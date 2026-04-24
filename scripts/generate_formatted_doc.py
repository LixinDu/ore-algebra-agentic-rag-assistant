#!/usr/bin/env python3
"""Generate formatted API docs for ore_algebra suitable for RAG ingestion."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Sequence


TRIPLE_QUOTE_START_RE = re.compile(r"^\s*[rRuUbBfF]*([\"']{3})")
CLASS_RE = re.compile(r"^(\s*)class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
DEF_RE = re.compile(r"^(\s*)def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*:")


@dataclass(frozen=True)
class SymbolRecord:
    id: str
    module: str
    qualname: str
    kind: str
    signature: str
    summary: str
    docstring: str
    example_count: int
    examples: List[str]
    file_path: str
    line: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create symbols.jsonl and API_REFERENCE.md from ore_algebra source."
    )
    parser.add_argument(
        "--input-root",
        default="src/ore_algebra",
        help="Input source root (default: src/ore_algebra).",
    )
    parser.add_argument(
        "--output-dir",
        default=".temp/ore_algebra_generated",
        help="Output directory (default: .temp/ore_algebra_generated).",
    )
    parser.add_argument(
        "--include-private",
        action="store_true",
        help="Include symbols starting with underscore.",
    )
    parser.add_argument(
        "--include-extensions",
        default=".py,.pyx,.spyx",
        help="Comma-separated list of source extensions to scan.",
    )
    return parser.parse_args()


def normalize_extensions(raw: str) -> List[str]:
    out = []
    for item in raw.split(","):
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        out.append(ext)
    if not out:
        raise ValueError("No valid extensions provided via --include-extensions")
    return sorted(set(out))


def should_include_symbol(name: str, include_private: bool) -> bool:
    return include_private or not name.startswith("_")


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def extract_sage_examples(docstring: str) -> List[str]:
    return [line.strip() for line in docstring.splitlines() if line.strip().startswith("sage:")]


def build_id(module: str, qualname: str) -> str:
    return f"{module}.{qualname}"


def signature_from_ast(node: ast.AST, default_name: str) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = ""
        return f"{default_name}({args})"
    if isinstance(node, ast.ClassDef):
        parts = []
        for base in node.bases:
            try:
                parts.append(ast.unparse(base))
            except Exception:
                parts.append("<?>")
        for kw in node.keywords:
            try:
                if kw.arg is None:
                    parts.append("**" + ast.unparse(kw.value))
                else:
                    parts.append(f"{kw.arg}={ast.unparse(kw.value)}")
            except Exception:
                if kw.arg is None:
                    parts.append("**<?>")
                else:
                    parts.append(f"{kw.arg}=<?>")
        if parts:
            return f"{default_name}({', '.join(parts)})"
        return default_name
    return default_name


def make_record(
    module: str,
    qualname: str,
    kind: str,
    signature: str,
    docstring: str,
    file_path: str,
    line: int,
) -> SymbolRecord:
    summary = first_nonempty_line(docstring)
    examples = extract_sage_examples(docstring)
    return SymbolRecord(
        id=build_id(module, qualname),
        module=module,
        qualname=qualname,
        kind=kind,
        signature=signature,
        summary=summary,
        docstring=docstring,
        example_count=len(examples),
        examples=examples,
        file_path=file_path,
        line=line,
    )


def gather_docstring_from_lines(lines: Sequence[str], start_index: int) -> str:
    i = start_index + 1
    n = len(lines)

    while i < n:
        stripped = lines[i].strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        break

    if i >= n:
        return ""

    m = TRIPLE_QUOTE_START_RE.match(lines[i])
    if not m:
        return ""

    quote = m.group(1)
    line = lines[i]
    start_pos = line.find(quote)
    remainder = line[start_pos + len(quote) :]
    end_pos = remainder.find(quote)
    if end_pos != -1:
        return remainder[:end_pos]

    collected = [remainder]
    i += 1
    while i < n:
        pos = lines[i].find(quote)
        if pos != -1:
            collected.append(lines[i][:pos])
            break
        collected.append(lines[i])
        i += 1
    return "\n".join(collected).strip("\n")


def compute_module_name(input_root: Path, path: Path) -> str:
    rel = path.relative_to(input_root)
    without_suffix = rel.as_posix()
    for suffix in (".py", ".pyx", ".spyx"):
        if without_suffix.endswith(suffix):
            without_suffix = without_suffix[: -len(suffix)]
            break
    return ".".join(part for part in without_suffix.split("/") if part)


def extract_from_py(
    path: Path, module: str, include_private: bool, display_path: str
) -> List[SymbolRecord]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    out: List[SymbolRecord] = []

    def visit_class(cls_node: ast.ClassDef, class_stack: List[str]) -> None:
        if should_include_symbol(cls_node.name, include_private):
            qualname = ".".join(class_stack + [cls_node.name])
            out.append(
                make_record(
                    module=module,
                    qualname=qualname,
                    kind="class",
                    signature=signature_from_ast(cls_node, cls_node.name),
                    docstring=ast.get_docstring(cls_node) or "",
                    file_path=display_path,
                    line=cls_node.lineno,
                )
            )
        next_stack = class_stack + [cls_node.name]
        for child in cls_node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not should_include_symbol(child.name, include_private):
                    continue
                qualname = ".".join(next_stack + [child.name])
                out.append(
                    make_record(
                        module=module,
                        qualname=qualname,
                        kind="method",
                        signature=signature_from_ast(child, child.name),
                        docstring=ast.get_docstring(child) or "",
                        file_path=display_path,
                        line=child.lineno,
                    )
                )
            elif isinstance(child, ast.ClassDef):
                visit_class(child, next_stack)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not should_include_symbol(node.name, include_private):
                continue
            out.append(
                make_record(
                    module=module,
                    qualname=node.name,
                    kind="function",
                    signature=signature_from_ast(node, node.name),
                    docstring=ast.get_docstring(node) or "",
                    file_path=display_path,
                    line=node.lineno,
                )
            )
        elif isinstance(node, ast.ClassDef):
            visit_class(node, [])

    return out


def extract_from_regex_fallback(
    path: Path, module: str, include_private: bool, display_path: str
) -> List[SymbolRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[SymbolRecord] = []
    class_stack: List[tuple[int, str]] = []

    for idx, line in enumerate(lines):
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        while class_stack and indent <= class_stack[-1][0]:
            class_stack.pop()

        class_match = CLASS_RE.match(line)
        if class_match:
            name = class_match.group(2)
            if should_include_symbol(name, include_private):
                qualname = ".".join([c[1] for c in class_stack] + [name])
                docstring = gather_docstring_from_lines(lines, idx)
                out.append(
                    make_record(
                        module=module,
                        qualname=qualname,
                        kind="class",
                        signature=name,
                        docstring=docstring,
                        file_path=display_path,
                        line=idx + 1,
                    )
                )
            class_stack.append((indent, name))
            continue

        def_match = DEF_RE.match(line)
        if def_match:
            name = def_match.group(2)
            if not should_include_symbol(name, include_private):
                continue
            args = def_match.group(3).strip()
            signature = f"{name}({args})"
            in_class = bool(class_stack)
            qual_parts = [c[1] for c in class_stack] + [name]
            kind = "method" if in_class else "function"
            docstring = gather_docstring_from_lines(lines, idx)
            out.append(
                make_record(
                    module=module,
                    qualname=".".join(qual_parts),
                    kind=kind,
                    signature=signature,
                    docstring=docstring,
                    file_path=display_path,
                    line=idx + 1,
                )
            )

    return out


def sort_symbols(symbols: Iterable[SymbolRecord]) -> List[SymbolRecord]:
    return sorted(symbols, key=lambda s: (s.module, s.qualname, s.kind, s.file_path, s.line))


def jsonl_for_symbols(symbols: Sequence[SymbolRecord]) -> str:
    lines = []
    for symbol in symbols:
        obj = {
            "id": symbol.id,
            "module": symbol.module,
            "qualname": symbol.qualname,
            "kind": symbol.kind,
            "signature": symbol.signature,
            "summary": symbol.summary,
            "docstring": symbol.docstring,
            "example_count": symbol.example_count,
            "examples": symbol.examples,
            "file_path": symbol.file_path,
            "line": symbol.line,
        }
        lines.append(json.dumps(obj, ensure_ascii=True, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def md_escape(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", "<br>")


def markdown_for_symbols(
    symbols: Sequence[SymbolRecord], module_count: int, generated_timestamp: str
) -> str:
    by_module = {}
    for symbol in symbols:
        by_module.setdefault(symbol.module, []).append(symbol)

    with_examples = sum(1 for s in symbols if s.example_count > 0)
    out = []
    out.append("# API Reference")
    out.append("")
    out.append(f"- Generated: `{generated_timestamp}`")
    out.append(f"- Modules scanned: `{module_count}`")
    out.append(f"- Symbols extracted: `{len(symbols)}`")
    out.append(f"- Symbols with examples: `{with_examples}`")
    out.append("")

    for module in sorted(by_module):
        out.append(f"## `{module}`")
        out.append("")
        out.append("| Symbol | Kind | Signature | Examples | Source | Summary |")
        out.append("| --- | --- | --- | ---: | --- | --- |")
        module_symbols = sorted(by_module[module], key=lambda s: (s.qualname, s.kind, s.line))
        for symbol in module_symbols:
            source = f"{symbol.file_path}:{symbol.line}"
            out.append(
                "| "
                + f"`{md_escape(symbol.qualname)}` | "
                + f"`{md_escape(symbol.kind)}` | "
                + f"`{md_escape(symbol.signature)}` | "
                + f"{symbol.example_count} | "
                + f"`{md_escape(source)}` | "
                + f"{md_escape(symbol.summary)} |"
            )
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def deterministic_generated_timestamp(files: Sequence[Path]) -> str:
    if not files:
        return "1970-01-01T00:00:00Z"
    latest_ns = max(path.stat().st_mtime_ns for path in files)
    latest_seconds = latest_ns / 1_000_000_000
    return datetime.fromtimestamp(latest_seconds, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent), newline=""
    ) as tmp:
        tmp.write(content)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def main() -> int:
    args = parse_args()
    try:
        extensions = normalize_extensions(args.include_extensions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    cwd = Path.cwd().resolve()

    if not input_root.exists() or not input_root.is_dir():
        print(f"error: input root does not exist or is not a directory: {input_root}", file=sys.stderr)
        return 2

    files = sorted(
        p
        for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    symbols: List[SymbolRecord] = []
    modules_scanned = set()
    parse_failures = 0

    for file_path in files:
        module = compute_module_name(input_root, file_path)
        modules_scanned.add(module)
        try:
            display_path = file_path.resolve().relative_to(cwd).as_posix()
        except ValueError:
            display_path = file_path.as_posix()

        try:
            if file_path.suffix.lower() == ".py":
                records = extract_from_py(
                    file_path, module, args.include_private, display_path
                )
            else:
                records = extract_from_regex_fallback(
                    file_path, module, args.include_private, display_path
                )
            symbols.extend(records)
        except SyntaxError as exc:
            parse_failures += 1
            print(f"warning: failed to parse {display_path}: {exc}", file=sys.stderr)
        except UnicodeDecodeError as exc:
            parse_failures += 1
            print(f"warning: failed to read {display_path}: {exc}", file=sys.stderr)

    if files and parse_failures == len(files):
        print("error: all candidate files failed to parse", file=sys.stderr)
        return 1

    symbols = sort_symbols(symbols)

    jsonl_content = jsonl_for_symbols(symbols)
    generated_timestamp = deterministic_generated_timestamp(files)
    md_content = markdown_for_symbols(
        symbols,
        module_count=len(modules_scanned),
        generated_timestamp=generated_timestamp,
    )

    jsonl_path = output_dir / "symbols.jsonl"
    md_path = output_dir / "API_REFERENCE.md"
    atomic_write(jsonl_path, jsonl_content)
    atomic_write(md_path, md_content)

    print(f"Wrote {len(symbols)} symbols")
    print(f"JSONL: {jsonl_path}")
    print(f"Markdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
