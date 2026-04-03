#!/usr/bin/env python3
"""Tests for the hardened Sage runtime module."""

from __future__ import annotations

import contextlib
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sage_runtime


def _resolve_sage_10_or_skip() -> str:
    sage_bin = os.getenv("SAGE_BIN", "sage")
    try:
        proc = subprocess.run(
            [sage_bin, "--version"],
            capture_output=True,
            check=False,
            timeout=10,
        )
    except FileNotFoundError as exc:
        raise unittest.SkipTest(f"Sage binary not found: {sage_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise unittest.SkipTest(f"Sage version check timed out: {sage_bin}") from exc

    version_text = (
        sage_runtime._decode_output(proc.stdout)
        + sage_runtime._decode_output(proc.stderr)
    ).strip()
    if proc.returncode != 0:
        raise unittest.SkipTest(f"Sage version check failed: {version_text or proc.returncode}")
    if "10.0" not in version_text:
        raise unittest.SkipTest(f"Sage 10.0 is required for integration tests, got: {version_text}")
    return sage_bin


class TestValidation(unittest.TestCase):
    def test_blocks_import_os(self) -> None:
        code = "import os\nA = OreAlgebra(QQ['x'], 'Dx')\n"
        with mock.patch.object(sage_runtime, "_preparse_with_sage", return_value=code):
            errors = sage_runtime.validate_generated_code(code)
        self.assertTrue(any("os" in error for error in errors), errors)

    def test_preparser_allows_sage_shorthand(self) -> None:
        sage_bin = _resolve_sage_10_or_skip()
        code = "R.<x> = QQ['x']\nA.<Dx> = OreAlgebra(R)\n"
        errors = sage_runtime.validate_generated_code(code, sage_bin=sage_bin)
        self.assertEqual(errors, [])

    def test_rejects_code_without_ore_algebra_marker(self) -> None:
        code = "print(2 + 2)\n"
        with mock.patch.object(sage_runtime, "_preparse_with_sage", return_value=code):
            errors = sage_runtime.validate_generated_code(code)
        self.assertTrue(any("ore_algebra marker" in error for error in errors), errors)


class TestSageExecutionIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sage_bin = _resolve_sage_10_or_skip()

    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        patcher = mock.patch.object(sage_runtime, "SCRIPT_LOG_DIR", Path(self._tempdir.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def _quiet_call(self, fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return fn(*args, **kwargs)

    def test_runs_simple_ore_algebra_script(self) -> None:
        code = (
            "A = OreAlgebra(QQ['x'], 'Dx')\n"
            "Dx = A.gen()\n"
            "print((Dx + 1) * (Dx + 2))\n"
        )
        result = self._quiet_call(
            sage_runtime.run_sage_code,
            code,
            sage_bin=self.sage_bin,
            timeout=60,
        )
        self.assertEqual(result.status, "success")
        self.assertTrue(result.preflight_ok)
        self.assertIn("Dx^2 + 3*Dx + 2", result.stdout_full)

    def test_blocks_import_os_before_execution(self) -> None:
        code = "import os\nA = OreAlgebra(QQ['x'], 'Dx')\n"
        result = self._quiet_call(
            sage_runtime.validate_and_run_sage,
            code,
            sage_bin=self.sage_bin,
            timeout=60,
        )
        self.assertEqual(result.status, "blocked")
        self.assertFalse(result.preflight_ok)
        self.assertTrue(any("os" in error for error in result.validation_errors), result.validation_errors)

    def test_times_out_on_infinite_loop(self) -> None:
        code = "A = OreAlgebra(QQ['x'], 'Dx')\nwhile True:\n    pass\n"
        result = self._quiet_call(
            sage_runtime.validate_and_run_sage,
            code,
            sage_bin=self.sage_bin,
            timeout=1,
        )
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.returncode, -124)
