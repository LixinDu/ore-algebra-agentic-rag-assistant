#!/usr/bin/env python3
"""Tests for llm_service staged code generation and execution-aware answers."""

from __future__ import annotations

import unittest
from unittest import mock

import llm_service
from sage_runtime import SageExecutionResult


def _contexts() -> list[llm_service.ContextItem]:
    return [
        llm_service.ContextItem(
            context_id="ctx_1",
            source_type="generated",
            title="OreAlgebra",
            location="generated/symbols.jsonl:1",
            text="OreAlgebra constructs operator algebras.",
            score=0.9,
        ),
        llm_service.ContextItem(
            context_id="ctx_2",
            source_type="generated",
            title="Operators",
            location="generated/symbols.jsonl:2",
            text="Dx is the derivation generator.",
            score=0.8,
        ),
    ]


class TestCodeGenerationResponse(unittest.TestCase):
    def test_build_code_generation_prompt_requires_printed_result(self) -> None:
        request = llm_service.CodeGenerationRequest(
            question="Compute and show LCLM",
            contexts=_contexts(),
            provider="openai",
            model="test-model",
        )
        prompt = llm_service.build_code_generation_prompt(request)
        self.assertIn("include explicit `print(...)` statements", prompt)

    def test_parse_code_generation_response_filters_citations(self) -> None:
        raw = (
            '{"code":"A = OreAlgebra(QQ[\'x\'], \'Dx\')",'
            '"citations_used":["ctx_1","ctx_999"],'
            '"missing_info":["none"]}'
        )
        response = llm_service.parse_code_generation_response(raw, ["ctx_1", "ctx_2"])
        self.assertEqual(response.code, "A = OreAlgebra(QQ['x'], 'Dx')")
        self.assertEqual(response.citations_used, ["ctx_1"])
        self.assertEqual(response.missing_info, ["none"])

    def test_generate_code_with_llm_repairs_invalid_json(self) -> None:
        request = llm_service.CodeGenerationRequest(
            question="Create ore_algebra code",
            contexts=_contexts(),
            provider="openai",
            model="test-model",
        )
        repaired = (
            '{"code":"A = OreAlgebra(QQ[\'x\'], \'Dx\')",'
            '"citations_used":["ctx_2"],'
            '"missing_info":[]}'
        )
        with mock.patch.object(llm_service, "_call_llm", side_effect=["not json", repaired]) as patched:
            response = llm_service.generate_code_with_llm(request=request)
        self.assertEqual(response.citations_used, ["ctx_2"])
        self.assertEqual(response.missing_info, [])
        self.assertEqual(patched.call_count, 2)


class TestExecutionAwareAnswer(unittest.TestCase):
    def test_parse_execution_answer_response_filters_citations(self) -> None:
        raw = (
            '{"answer":"The result is $Dx^2 + 3Dx + 2$.",'
            '"citations_used":["ctx_2","ctx_missing"],'
            '"missing_info":[]}'
        )
        response = llm_service.parse_execution_answer_response(raw, ["ctx_1", "ctx_2"])
        self.assertEqual(response.answer, "The result is $Dx^2 + 3Dx + 2$.")
        self.assertEqual(response.citations_used, ["ctx_2"])

    def test_parse_execution_answer_response_handles_unescaped_latex_backslashes(self) -> None:
        raw = (
            '{"answer":"Computed $\\left(Dx + 1\\right)\\left(Dx + 2\\right)$.",'
            '"citations_used":["ctx_1"],'
            '"missing_info":[]}'
        )
        response = llm_service.parse_execution_answer_response(raw, ["ctx_1", "ctx_2"])
        self.assertIn("\\left", response.answer)
        self.assertEqual(response.citations_used, ["ctx_1"])

    def test_answer_with_execution_llm_repairs_invalid_json(self) -> None:
        request = llm_service.ExecutionAwareAnswerRequest(
            question="What happened?",
            contexts=_contexts(),
            original_code="A = OreAlgebra(QQ['x'], 'Dx')",
            execution_result=SageExecutionResult(
                status="success",
                preflight_ok=True,
                stdout_full="Dx^2 + 3*Dx + 2\n",
                stdout_summary="Dx^2 + 3*Dx + 2\n",
                is_truncated=False,
                stderr="",
                returncode=0,
                validation_errors=[],
            ),
            code_generation_citations=["ctx_1"],
            provider="openai",
            model="test-model",
        )
        repaired = (
            '{"answer":"The computed result is $Dx^2 + 3Dx + 2$.",'
            '"citations_used":["ctx_1"],'
            '"missing_info":[]}'
        )
        with mock.patch.object(llm_service, "_call_llm", side_effect=["bad", repaired]) as patched:
            response = llm_service.answer_with_execution_llm(request=request)
        self.assertIn("$Dx^2 + 3Dx + 2$", response.answer)
        self.assertEqual(response.citations_used, ["ctx_1"])
        self.assertEqual(patched.call_count, 2)

    def test_build_execution_answer_prompt_mentions_truncation(self) -> None:
        request = llm_service.ExecutionAwareAnswerRequest(
            question="Summarize the computation",
            contexts=_contexts(),
            original_code="print('hello')",
            execution_result=SageExecutionResult(
                status="success",
                preflight_ok=True,
                stdout_full="x" * 5000,
                stdout_summary="visible summary",
                is_truncated=True,
                stderr="",
                returncode=0,
                validation_errors=[],
            ),
            code_generation_citations=["ctx_1"],
        )
        prompt = llm_service.build_execution_answer_prompt(request)
        self.assertIn("Full Output", prompt)
        self.assertIn("is_truncated: true", prompt)
        self.assertIn("visible summary", prompt)
        self.assertIn("include the concrete computed runtime result", prompt)
        self.assertIn("Escape every LaTeX backslash", prompt)

    def test_build_execution_answer_prompt_handles_skipped_execution(self) -> None:
        request = llm_service.ExecutionAwareAnswerRequest(
            question="Summarize the computation",
            contexts=_contexts(),
            original_code="",
            execution_result=None,
            execution_skipped_reason="No code generated; execution skipped.",
            code_generation_citations=[],
        )
        prompt = llm_service.build_execution_answer_prompt(request)
        self.assertIn("status: skipped", prompt)
        self.assertIn("No code generated; execution skipped.", prompt)


if __name__ == "__main__":
    unittest.main()
