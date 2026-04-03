#!/usr/bin/env python3
"""Hardened SageMath execution helpers for ore_algebra code."""

from __future__ import annotations

import ast
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4


OUTPUT_SUMMARY_MAX_CHARS = 2000
OUTPUT_TRUNCATION_HINT = (
    "[Output Truncated] ... Total length: {length} chars. "
    "Suggest querying specific properties like .degree() or .coefficient()."
)
BLOCKED_IMPORT_ROOTS = {"os", "subprocess", "shutil", "socket", "requests", "urllib"}
BLOCKED_CALLS = {"eval", "exec", "__import__"}
ORE_ALGEBRA_MARKERS = ("OreAlgebra", "ore_algebra")
SCRIPT_LOG_DIR = Path(__file__).resolve().parent / "scripts_log"
AUTO_IMPORTS = "from sage.all import *\nfrom ore_algebra import *\n"


@dataclass
class SageExecutionResult:
    status: Literal["success", "error", "blocked", "timeout"]
    preflight_ok: bool
    stdout_full: str
    stdout_summary: str
    is_truncated: bool
    stderr: str
    returncode: int
    validation_errors: list[str] = field(default_factory=list)


def _effective_sage_bin(sage_bin: str) -> str:
    explicit = (sage_bin or "").strip()
    env_override = os.getenv("SAGE_BIN", "").strip()
    if explicit and explicit != "sage":
        return explicit
    if env_override:
        return env_override
    return explicit or "sage"


def _decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _append_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _blocked_import_name(name: str) -> str | None:
    root = (name or "").split(".", 1)[0]
    if root in BLOCKED_IMPORT_ROOTS:
        return root
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _summarize_stdout(stdout_text: str, max_chars: int = OUTPUT_SUMMARY_MAX_CHARS) -> tuple[str, bool]:
    if len(stdout_text) <= max_chars:
        return stdout_text, False

    suffix = OUTPUT_TRUNCATION_HINT.format(length=len(stdout_text))
    if len(suffix) >= max_chars:
        return suffix[:max_chars], True

    head = stdout_text[: max_chars - len(suffix)]
    return f"{head}{suffix}", True


def _result(
    *,
    status: Literal["success", "error", "blocked", "timeout"],
    preflight_ok: bool,
    stdout_full: str = "",
    stderr: str = "",
    returncode: int = 0,
    validation_errors: list[str] | None = None,
) -> SageExecutionResult:
    stdout_summary, is_truncated = _summarize_stdout(stdout_full)
    return SageExecutionResult(
        status=status,
        preflight_ok=preflight_ok,
        stdout_full=stdout_full,
        stdout_summary=stdout_summary,
        is_truncated=is_truncated,
        stderr=stderr,
        returncode=returncode,
        validation_errors=list(validation_errors or []),
    )


def _script_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return SCRIPT_LOG_DIR / f"sage_exec_{timestamp}_{uuid4().hex[:8]}.py"


def _preparse_with_sage(code: str, sage_bin: str) -> str:
    effective_bin = _effective_sage_bin(sage_bin)
    helper = (
        "import sys\n"
        "from sage.all import preparse\n"
        "sys.stdout.write(preparse(sys.stdin.read()))\n"
    )
    try:
        proc = subprocess.run(
            [effective_bin, "-python", "-c", helper],
            input=code.encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Sage binary not found: {effective_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Sage preparser timed out after 60 seconds using {effective_bin}"
        ) from exc

    if proc.returncode != 0:
        stderr_text = _decode_output(proc.stderr).strip()
        detail = stderr_text or f"exit code {proc.returncode}"
        raise RuntimeError(f"Sage preparser failed: {detail}")

    return _decode_output(proc.stdout)


def validate_generated_code(code: str, sage_bin: str = "sage") -> list[str]:
    errors: list[str] = []
    stripped = code.strip()
    if not stripped:
        return ["Code is empty."]

    if not any(marker in code for marker in ORE_ALGEBRA_MARKERS):
        _append_error(
            errors,
            "Code must contain an ore_algebra marker (`OreAlgebra` or `ore_algebra`).",
        )

    try:
        preparsed_code = _preparse_with_sage(code, sage_bin=sage_bin)
    except Exception as exc:
        _append_error(errors, str(exc))
        return errors

    try:
        tree = ast.parse(preparsed_code)
    except SyntaxError as exc:
        detail = exc.msg
        if exc.lineno is not None:
            detail = f"{detail} (line {exc.lineno})"
        _append_error(errors, f"Preparsed code is not valid Python: {detail}")
        return errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                blocked = _blocked_import_name(alias.name)
                if blocked is not None:
                    _append_error(errors, f"Blocked import detected: {blocked}")
        elif isinstance(node, ast.ImportFrom):
            blocked = _blocked_import_name(node.module or "")
            if blocked is not None:
                _append_error(errors, f"Blocked import detected: {blocked}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in BLOCKED_CALLS:
                _append_error(errors, f"Blocked builtin call detected: {call_name}()")

    return errors


def run_sage_code(code: str, sage_bin: str = "sage", timeout: int = 60) -> SageExecutionResult:
    effective_bin = _effective_sage_bin(sage_bin)
    final_script = f"{AUTO_IMPORTS}{code.strip()}\n"

    print("=== Sage code to execute ===")
    print(code.rstrip())

    SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = _script_path()
    script_path.write_text(final_script, encoding="utf-8")

    should_delete_script = False
    try:
        preparsed_script = _preparse_with_sage(final_script, sage_bin=effective_bin)
        script_path.write_text(preparsed_script, encoding="utf-8")
        proc = subprocess.run(
            [effective_bin, "-python", str(script_path)],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        stdout_text = _decode_output(proc.stdout)
        stderr_text = _decode_output(proc.stderr)
        status: Literal["success", "error"] = "success" if proc.returncode == 0 else "error"
        if status == "success":
            should_delete_script = True
        else:
            print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status=status,
            preflight_ok=True,
            stdout_full=stdout_text,
            stderr=stderr_text,
            returncode=proc.returncode,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = _decode_output(exc.stdout)
        stderr_text = _decode_output(exc.stderr)
        timeout_message = f"Execution timed out after {timeout} seconds."
        if stderr_text:
            stderr_text = f"{stderr_text.rstrip()}\n{timeout_message}"
        else:
            stderr_text = timeout_message
        print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status="timeout",
            preflight_ok=True,
            stdout_full=stdout_text,
            stderr=stderr_text,
            returncode=-124,
        )
    except FileNotFoundError:
        print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status="error",
            preflight_ok=True,
            stderr=f"Sage binary not found: {effective_bin}",
            returncode=127,
        )
    except Exception as exc:
        print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status="error",
            preflight_ok=True,
            stderr=str(exc),
            returncode=1,
        )
    finally:
        if should_delete_script:
            try:
                script_path.unlink()
            except OSError:
                pass


def validate_and_run_sage(code: str, sage_bin: str = "sage", timeout: int = 60) -> SageExecutionResult:
    validation_errors = validate_generated_code(code, sage_bin=sage_bin)
    if validation_errors:
        return _result(
            status="blocked",
            preflight_ok=False,
            stderr="Code failed validation.",
            returncode=-1,
            validation_errors=validation_errors,
        )
    return run_sage_code(code, sage_bin=sage_bin, timeout=timeout)
