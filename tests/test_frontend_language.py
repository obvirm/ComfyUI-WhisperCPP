"""Frontend checks for issue #16, run under Node.js.

Both are skipped when Node is not installed; the backend side is covered by
``test_languages.py`` and ``test_issue16_validation.py``.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
JS_FILE = REPO_ROOT / "js" / "whispercpp_node.js"
SCRIPT = Path(__file__).resolve().parent / "check_lang_sanitize.mjs"
NODE = shutil.which("node")


def _run(*args):
    return subprocess.run(
        list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(REPO_ROOT),
    )


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_node_file_parses_as_module():
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "whispercpp_node.mjs"
        shutil.copyfile(JS_FILE, copy)
        res = _run(NODE, "--check", str(copy))
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_frontend_language_sanitizer():
    res = _run(NODE, str(SCRIPT))
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "OK" in res.stdout, res.stdout
