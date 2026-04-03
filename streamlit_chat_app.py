#!/usr/bin/env python3
"""Streamlit agent app: plan -> retrieve per step -> decide -> final synthesis."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import List

import streamlit as st

from llm_service import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    ContextItem,
    ExecutionAwareAnswerRequest,
    FinalAnswerResponse,
    answer_with_execution_llm,
    Subtask,
    decide_next_action,
    generate_code_with_llm,
    list_ollama_models,
    plan_subtasks,
)
from ore_rag_assistant import (
    RetrievalResult,
    load_index,
    location_label,
    parse_chunks,
    select_retrieval,
)
from sage_runtime import SageExecutionResult, validate_and_run_sage


def _result_title(result: RetrievalResult) -> str:
    if result.source_type == "generated":
        name = result.qualname or result.symbol_id or "unknown_symbol"
        return f"{name} [{result.score:.4f}]"
    pages = (
        f"{result.page_start}-{result.page_end}"
        if result.page_start is not None and result.page_end is not None
        else str(result.page_start or "?")
    )
    return f"{result.source} pp.{pages} [{result.score:.4f}]"


def _to_context_items(results: List[RetrievalResult], pdf_char_limit: int) -> List[ContextItem]:
    items: List[ContextItem] = []
    for i, r in enumerate(results, start=1):
        context_id = f"ctx_{i}"
        if r.source_type == "generated":
            title = r.qualname or r.symbol_id or "unknown_symbol"
            text = r.text
        else:
            title = r.section_title or r.source
            text = r.text[:pdf_char_limit]
        items.append(
            ContextItem(
                context_id=context_id,
                source_type=r.source_type,
                title=title,
                location=location_label(r),
                text=text,
                score=r.score,
            )
        )
    return items


def _dedupe_results_by_chunk(results: List[RetrievalResult]) -> List[RetrievalResult]:
    seen = set()
    out: List[RetrievalResult] = []
    for r in results:
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        out.append(r)
    return out


def _citation_lines(citations_used: List[str], context_items: List[ContextItem]) -> str:
    ctx_by_id = {c.context_id: c for c in context_items}
    lines: List[str] = []
    for cid in citations_used:
        item = ctx_by_id.get(cid)
        if not item:
            continue
        lines.append(f"- {cid}: {item.title} ({item.location})")
    if not lines:
        return "No citations returned."
    return "\n".join(lines)


def _order_contexts_cited_first(
    context_items: List[ContextItem],
    cited_context_ids: List[str],
) -> List[ContextItem]:
    if not context_items or not cited_context_ids:
        return list(context_items)

    cited_ids = set(cited_context_ids)
    cited_items: List[ContextItem] = []
    uncited_items: List[ContextItem] = []
    for item in context_items:
        if item.context_id in cited_ids:
            cited_items.append(item)
        else:
            uncited_items.append(item)
    return cited_items + uncited_items


def _download_link(data: str, filename: str, label: str = "Download") -> str:
    b64 = base64.b64encode(data.encode()).decode()
    return (
        f'<a href="data:text/plain;base64,{b64}" download="{filename}" '
        f'style="text-decoration:none;font-size:0.85em;">{label}</a>'
    )


def _wrap_latex_envs(text: str) -> str:
    # Normalize literal \n sequences to real newlines
    text = text.replace('\\n', '\n')
    # Wrap block LaTeX environments with $$
    text = re.sub(
        r'(\\begin\{[^}]+\}.*?\\end\{[^}]+\})',
        r'$$\1$$',
        text,
        flags=re.DOTALL,
    )
    return text


def _wrap_inline_latex(text: str) -> str:
    # Wrap inline LaTeX commands like \texttt{...}, \text{...} with $...$
    return re.sub(r'(\\[a-zA-Z]+\{[^}]*\})', r'$\1$', text)


def _render_answer(text: str) -> None:
    text = _wrap_latex_envs(text)
    parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            st.latex(part[2:-2].strip())
        elif part.strip():
            part = re.sub(r'\\+\s*$', '', part)
            st.markdown(_wrap_inline_latex(part))


def _execution_status_label(execution_result: SageExecutionResult | None) -> str:
    if execution_result is None:
        return "skipped"
    return execution_result.status



def _execution_detail_text(
    execution_result: SageExecutionResult | None,
    skipped_reason: str,
) -> str:
    if execution_result is None:
        return skipped_reason.strip() or "Execution was skipped."

    detail_parts: List[str] = []
    if execution_result.validation_errors:
        detail_parts.append("Validation errors:")
        detail_parts.extend(f"- {item}" for item in execution_result.validation_errors)
    if execution_result.stderr.strip():
        if detail_parts:
            detail_parts.append("")
        detail_parts.append("stderr:")
        detail_parts.append(execution_result.stderr.rstrip())
    if not detail_parts:
        return "(No stderr or validation errors.)"
    return "\n".join(detail_parts)


def _render_retrieval_results(results: List[RetrievalResult], pdf_char_limit: int, key_prefix: str) -> None:
    st.subheader("Retrieved Context")
    for idx, r in enumerate(results):
        with st.expander(_result_title(r), expanded=False):
            st.write(f"source_type: `{r.source_type}`")
            st.write(f"source: `{r.source}`")
            if r.source_type == "generated":
                if r.signature:
                    st.write(f"signature: `{r.signature}`")
                if r.module:
                    st.write(f"module: `{r.module}`")
                st.write(f"location: `{location_label(r)}`")
                st.text_area(
                    "content",
                    value=r.text,
                    height=260,
                    key=f"text-{key_prefix}-{idx}-{r.chunk_id}-gen",
                )
            else:
                section = r.section_title or "unknown"
                st.write(f"pages: `{r.page_start}-{r.page_end}`")
                st.write(f"section: `{section}`")
                preview = r.text[:pdf_char_limit] if len(r.text) > pdf_char_limit else r.text
                st.text_area(
                    "content",
                    value=preview,
                    height=260,
                    key=f"text-{key_prefix}-{idx}-{r.chunk_id}-pdf",
                )


def _run_retrieval_for_query(
    *,
    query: str,
    payload: dict,
    chunks: list,
    k: int,
    mode: str,
    index_path: Path,
    hybrid_alpha: float,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
) -> tuple[str, List[RetrievalResult]]:
    return select_retrieval(
        index_payload=payload,
        chunks=chunks,
        query=query,
        k=k,
        mode=mode,
        index_path=index_path,
        hybrid_alpha=hybrid_alpha,
        source_priority=source_priority,
        symbols_ratio=symbols_ratio,
        max_pdf_extras=max_pdf_extras,
    )


def _render_step_header(step: Subtask, step_query: str) -> None:
    st.markdown(f"### Step {step.step_id}: {step.title}")
    st.write(f"instruction: {step.instruction}")
    st.write(f"query: `{step_query}`")


@st.cache_data(ttl=5)
def _cached_ollama_models(base_url: str) -> tuple[list[str], str]:
    return list_ollama_models(base_url=base_url)


def main() -> None:
    st.set_page_config(page_title="ore_algebra Assistant", layout="wide", initial_sidebar_state="collapsed",)
    st.title("ore_algebra Assistant")

    with st.sidebar:
        st.header("Knowledge Base")
        index_path = st.text_input("Index path", ".rag/ore_algebra_both_index.json")
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("Top-k per step", min_value=1, max_value=20, value=6, help="Retrieved results per planning step.")
        with col2:
            final_context_limit = st.slider("Final Top-k", min_value=2, max_value=30, value=10, help="Context chunks passed to the final answer LLM.")

        with st.expander("Advanced search settings"):
            mode = st.selectbox("Search mode", ["auto", "hybrid", "dense", "lexical"], index=0)
            hybrid_alpha = st.slider("Hybrid weight", min_value=0.0, max_value=1.0, value=0.7, help="1.0 = dense only, 0.0 = lexical only.")
            source_priority = st.selectbox("Source preference", ["auto", "symbols-first", "flat"], index=0)
            symbols_ratio = st.slider("Symbol vs text ratio", min_value=0.0, max_value=1.0, value=0.75)
            max_pdf_extras = st.slider("Max PDF results", min_value=0, max_value=10, value=2)
            pdf_char_limit = st.slider("PDF chunk length", min_value=400, max_value=3000, value=1600)

        st.header("Language Model")
        provider = st.selectbox("Provider", ["openai", "gemini", "ollama"], index=2)
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="Lower = more deterministic, higher = more creative.",
        )
        api_key = ""
        llm_base_url: str | None = None

        if provider == "openai":
            model = st.text_input("Model", "gpt-4o-mini")
            api_key = st.text_input("API key", type="password")
            has_default_key = bool(os.getenv("OPENAI_API_KEY"))
            st.caption("Using key from environment." if (not api_key and has_default_key) else "" if api_key else "No API key found.")
        elif provider == "gemini":
            model = st.text_input("Model", "gemini-2.5-flash")
            api_key = st.text_input("API key", type="password")
            has_default_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            st.caption("Using key from environment." if (not api_key and has_default_key) else "" if api_key else "No API key found.")
        else:
            llm_base_url = st.text_input(
                "Ollama base URL",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ).strip()
            detected_models, detect_error = _cached_ollama_models(llm_base_url or "http://localhost:11434")
            if detected_models:
                default_model = "qwen3-coder:30b"
                default_index = detected_models.index(default_model) if default_model in detected_models else 0
                selected_model = st.selectbox("Model", detected_models, index=default_index)
                custom_model = st.text_input("Override model (optional)", "")
                model = custom_model.strip() or selected_model
                st.caption(f"{len(detected_models)} model(s) detected locally.")
            else:
                model = st.text_input("Model", os.getenv("OLLAMA_MODEL", "llama3.1"))
                st.caption(f"No models detected. {detect_error}" if detect_error else "No models detected. Run `ollama pull <model>`.")

        st.header("Sage Execution")
        sage_bin = st.text_input("Sage path", os.getenv("SAGE_BIN", "sage"))
        col3, col4 = st.columns(2)
        with col3:
            execution_timeout = st.slider("Timeout (s)", min_value=1, max_value=180, value=60)
        with col4:
            max_plan_steps = st.slider("Max steps", min_value=1, max_value=10, value=4, help="Maximum planning steps.")

    question = st.chat_input("Ask a question about ore_algebra...")
    if question:
        st.session_state["last_question"] = question

    if not question:
        last_question = st.session_state.get("last_question", "")
        if last_question:
            st.subheader("Last Question")
            st.markdown(last_question)
        st.info("You can ask about symbolic computations in ore_algebra.")
        return

    with st.chat_message("user"):
        st.markdown(question)

    idx = Path(index_path).expanduser().resolve()
    if not idx.exists():
        st.error(f"Index file not found: {idx}")
        return

    try:
        with st.spinner("Loading index..."):
            payload = load_index(idx)
            chunks = parse_chunks(payload)
    except Exception as exc:
        st.error(f"Index load failed: {exc}")
        return

    try:
        with st.spinner("Planning subtasks..."):
            plan = plan_subtasks(
                question=question,
                provider=provider,
                model=model,
                api_key=api_key or None,
                base_url=llm_base_url,
                max_steps=max_plan_steps,
                temperature=temperature,
            )
    except Exception as exc:
        st.error(f"Planning failed: {exc}")
        return

    st.subheader("Solution Plan")
    if not plan.subtasks:
        st.warning("Planner returned no subtasks.")
        return

    aggregated_results: List[RetrievalResult] = []

    for subtask in plan.subtasks:
        
        step_query = subtask.retrieval_query or subtask.instruction or subtask.title
       
        #st.subheader(f"Step {subtask.step_id} — {subtask.title}",expanded=False)
        with st.status(f"Step {subtask.step_id} — {subtask.title}", expanded=False) as step_status:
            st.write(f"instruction: {subtask.instruction}")
            st.write(f"query: {step_query}")

            mode_used, results = _run_retrieval_for_query(
                query=step_query,
                payload=payload,
                chunks=chunks,
                k=k,
                mode=mode,
                index_path=idx,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
            )
            st.write(f"Search mode: `{mode_used}`, results: `{len(results)}`")
            _render_retrieval_results(
                results=results,
                pdf_char_limit=pdf_char_limit,
                key_prefix=f"step-{subtask.step_id}-base",
            )
            context_items = _to_context_items(results=results, pdf_char_limit=pdf_char_limit)

            decision = decide_next_action(
                question=question,
                current_step=subtask,
                context_items=context_items,
                provider=provider,
                model=model,
                api_key=api_key or None,
                base_url=llm_base_url,
                temperature=temperature,
            )

            st.write(f"Next step: `{decision.action}`")
            if decision.reason:
                st.write(f"reason: {decision.reason}")
            st.write(f"confidence: {decision.confidence:.2f}")
            
            step_effective_results = results
            if decision.action == "refine_query" and decision.next_query.strip():
                st.write(f"refine query: `{decision.next_query}`")
                mode_used2, refined_results = _run_retrieval_for_query(
                    query=decision.next_query.strip(),
                    payload=payload,
                    chunks=chunks,
                    k=k,
                    mode=mode,
                    index_path=idx,
                    hybrid_alpha=hybrid_alpha,
                    source_priority=source_priority,
                    symbols_ratio=symbols_ratio,
                    max_pdf_extras=max_pdf_extras,
                )
                st.write(f"refined retrieval mode: `{mode_used2}`, results: `{len(refined_results)}`")
                _render_retrieval_results(
                    results=refined_results,
                    pdf_char_limit=pdf_char_limit,
                    key_prefix=f"step-{subtask.step_id}-refined",
                )
                step_effective_results = refined_results

            aggregated_results.extend(step_effective_results)

            if decision.action == "stop":
                st.info("Workflow stopped by agent decision.")
                break

            step_status.update(
                state="complete",
                expanded=False,
            )


    aggregated_results = _dedupe_results_by_chunk(aggregated_results)[:final_context_limit]
    if not aggregated_results:
        st.warning("No aggregated context available for final synthesis.")
        return

    final_context_items = _to_context_items(
        results=aggregated_results,
        pdf_char_limit=pdf_char_limit,
    )
    code_request = CodeGenerationRequest(
        question=question,
        contexts=final_context_items,
        provider=provider,
        model=model,
        temperature=temperature,
        base_url=llm_base_url,
    )

    code_response: CodeGenerationResponse
    execution_result: SageExecutionResult | None = None
    execution_skipped_reason = ""
    final_response: FinalAnswerResponse

    try:
        st.divider()
        st.subheader("Computation")
        with st.status("Sage Code", expanded=False) as code_status:
            code_response = generate_code_with_llm(
                request=code_request,
                api_key=api_key or None,
            )

            if code_response.code.strip():
                st.code(code_response.code, language="python")
            else:
                st.caption("No code generated.")

            citations_text = _citation_lines(code_response.citations_used, final_context_items)
            st.caption("Citations")
            st.markdown(citations_text)

            if code_response.missing_info:
                st.caption("Missing info")
                st.markdown("\n".join(f"- {item}" for item in code_response.missing_info))

            with st.expander("Raw output", expanded=False):
                st.code(code_response.raw_response or "(No raw response captured)", language="json")

            code_status.update(label="Sage Code", state="complete", expanded=False)

        with st.status("Sage Execution", expanded=False) as exec_status:
            if code_response.code.strip():
                execution_result = validate_and_run_sage(
                    code_response.code,
                    sage_bin=sage_bin,
                    timeout=execution_timeout,
                )
            else:
                execution_skipped_reason = (
                    "No code was generated from the retrieved context, so Sage execution was skipped."
                )

            status_label = _execution_status_label(execution_result)
            if status_label == "success":
                st.success("success")
            elif status_label == "skipped":
                st.warning("skipped")
            else:
                st.error(status_label)

            if execution_result is not None:
                meta = [
                    f"returncode: {execution_result.returncode}",
                    f"preflight: {'ok' if execution_result.preflight_ok else 'failed'}",
                ]
                if execution_result.is_truncated:
                    meta.append("output truncated by Sage")
                st.caption("  |  ".join(meta))

            has_errors = execution_result is not None and bool(
                execution_result.stderr.strip() or execution_result.validation_errors
            )
            tab_output, tab_errors = st.tabs(["Output", "Errors ⚠" if has_errors else "Errors"])

            with tab_output:
                if execution_result is None:
                    st.caption(execution_skipped_reason or "Execution was skipped.")
                else:
                    stdout_full = execution_result.stdout_full or ""
                    display_text = (execution_result.stdout_summary or "").strip() or stdout_full

                    if not display_text.strip():
                        if execution_result.status == "success":
                            st.caption("Execution succeeded but produced no output. Add `print(...)` in generated code.")
                        else:
                            st.caption("(No output available.)")
                    else:
                        line_limit = 40
                        lines = display_text.splitlines()
                        is_long = len(lines) > line_limit

                        st.code("\n".join(lines[:line_limit]) if is_long else display_text, language="text")

                        info = f"Showing {line_limit} of {len(lines)} lines." if is_long else ""
                        dl = _download_link(stdout_full, "sage_output.txt", "↓ Download full output") if stdout_full.strip() else ""
                        st.markdown(
                            f'<div style="display:flex;justify-content:space-between;align-items:center">'
                            f'<span style="font-size:0.8em;color:gray">{info}</span>{dl}</div>',
                            unsafe_allow_html=True,
                        )

                        if is_long:
                            with st.expander("Show full output"):
                                st.code(display_text, language="text")

            with tab_errors:
                if has_errors:
                    st.code(_execution_detail_text(execution_result, execution_skipped_reason), language="text")
                else:
                    st.caption("No errors.")

            exec_status.update(state="complete", expanded=False)

        st.divider()
        st.subheader("Final Answer")

        live_placeholder = st.empty()

        def _on_chunk(_piece: str, _acc_text: str) -> None:
            live_placeholder.status("Composing answer...", expanded=False)

        answer_request = ExecutionAwareAnswerRequest(
            question=question,
            contexts=_order_contexts_cited_first(
                final_context_items,
                code_response.citations_used,
            ),
            original_code=code_response.code,
            execution_result=execution_result,
            execution_skipped_reason=execution_skipped_reason,
            code_generation_citations=code_response.citations_used,
            provider=provider,
            model=model,
            temperature=temperature,
            base_url=llm_base_url,
        )

        final_response = answer_with_execution_llm(
            request=answer_request,
            api_key=api_key or None,
            stream=True,
            on_chunk=_on_chunk,
        )
        live_placeholder.empty()

        with st.chat_message("assistant"):
            _render_answer(final_response.answer or "(No answer returned)")

        with st.expander("Details", expanded=False):
            st.caption("Citations used")
            st.markdown(_citation_lines(final_response.citations_used, final_context_items))

            if final_response.missing_info:
                st.caption("Missing info")
                st.markdown("\n".join(f"- {item}" for item in final_response.missing_info))

            with st.expander("Raw output", expanded=False):
                st.code(final_response.raw_response or "(No raw response captured)", language="json")

        #stage_placeholder.write=st.empty()
    except Exception as exc:
        st.error(f"Code generation / Sage execution / final answer failed: {exc}")
        return


if __name__ == "__main__":
    main()
