"""ONNX Runtime pairing guard for sherpa-onnx alignment (issue #17).

Issue #17: ``align=true`` crashed ComfyUI itself (reconnect, no Python
traceback) with::

    The requested API version [28] is not available, only API versions [1, 27]
    are supported in this build. Current ORT Version: 1.27.0
    Windows fatal exception: access violation

Root cause: the ``_sherpa_onnx`` native module is built against a newer ONNX
Runtime API than the ``onnxruntime.dll`` it actually loads (missing or skewed
bundle inside ``sherpa_onnx/lib``). The native GetApi() returns NULL and the
next dereference is an uncatchable access violation.

The guard reads the bundled DLL's FileVersion in Python and raises
``AlignmentUnavailableError`` instead, which the node catches and turns into a
skipped alignment. These tests prove the guard on the real installed pairing
and its detection of skewed/missing bundles.
"""
import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_PKG = "whisperxx_issue17_under_test"
if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [str(REPO_ROOT)]
    sys.modules[_PKG] = _pkg

asr = importlib.import_module(f"{_PKG}.whispercpp.ext.alignment_sherpa")

sherpa_onnx = pytest.importorskip("sherpa_onnx", reason="sherpa-onnx not installed")


def test_parse_so_version():
    assert asr.parse_so_version("libonnxruntime.so.1.28.2") == (1, 28, 2, 0)
    assert asr.parse_so_version("libonnxruntime.so.1.24") is None
    assert asr.parse_so_version("libonnxruntime.so") is None
    assert asr.parse_so_version("libsherpa-onnx-c-api.so") is None


def test_preflight_detects_already_loaded_old_ort(monkeypatch):
    """A stale onnxruntime.dll pre-loaded in the process is the issue #17 trigger."""
    version = getattr(sherpa_onnx, "__version__", "")
    expected = asr._KNOWN_BUNDLED_ORT.get(version)
    if expected is None:
        pytest.skip(f"sherpa-onnx {version} not in known-release table")
    monkeypatch.setattr(
        asr, "loaded_ort_modules",
        lambda: [(r"C:\stale\onnxruntime.dll", (1, 0, 0, 0))],
    )
    with pytest.raises(asr.AlignmentUnavailableError, match="already loaded"):
        asr.preflight_ort()


def test_pick_newest_prefers_higher_ort(tmp_path):
    ort_util = importlib.import_module(_PKG + ".whispercpp.ort_util")
    a = tmp_path / "old.dll"
    b = tmp_path / "new.dll"
    a.write_bytes(b"x")
    b.write_bytes(b"x")
    versions = {a: (1, 27, 0, 0), b: (1, 28, 2, 0)}
    picked = ort_util.pick_newest_ort([a, b, tmp_path / "missing.dll"],
                                      version_fn=lambda p: versions.get(Path(p)))
    assert picked == b
    # unknown version ranks below everything
    c = tmp_path / "unknown.dll"
    c.write_bytes(b"x")
    picked = ort_util.pick_newest_ort([c, a], version_fn=lambda p: versions.get(Path(p)))
    assert picked == a


def test_bundled_ort_version_matches_known_release():
    version = getattr(sherpa_onnx, "__version__", "")
    bundled = asr.bundled_ort_version()
    assert bundled is not None, "bundled onnxruntime not found - install sherpa-onnx-core"
    expected = asr._KNOWN_BUNDLED_ORT.get(version)
    if expected is None:
        pytest.skip(f"sherpa-onnx {version} not in known-release table")
    assert bundled == expected, (
        f"sherpa-onnx {version} should bundle onnxruntime {expected}, found {bundled}"
    )


def test_preflight_passes_on_healthy_env():
    bundled = asr.preflight_ort()
    assert isinstance(bundled, tuple) and len(bundled) == 4


def test_preflight_detects_skewed_bundle(monkeypatch):
    """A bundle older than the release expects must fail cleanly (issue #17)."""
    version = getattr(sherpa_onnx, "__version__", "")
    table = dict(asr._KNOWN_BUNDLED_ORT)
    table[version] = (1, 99, 0, 0)  # pretend this release needs ORT 1.99
    monkeypatch.setattr(asr, "_KNOWN_BUNDLED_ORT", table)
    monkeypatch.setattr(asr, "loaded_ort_modules", lambda: [])
    with pytest.raises(asr.AlignmentUnavailableError, match="mismatched"):
        asr.preflight_ort()


def test_preflight_detects_missing_bundle(monkeypatch):
    monkeypatch.setattr(asr, "bundled_ort_version", lambda: None)
    with pytest.raises(asr.AlignmentUnavailableError, match="missing its bundled"):
        asr.preflight_ort()


def test_alignment_error_is_catchable_python_exception():
    """The node's `except Exception` must catch it - unlike an AV."""
    assert issubclass(asr.AlignmentUnavailableError, Exception)
    assert issubclass(asr.AlignmentUnavailableError, RuntimeError)


def test_load_align_model_runs_preflight_before_native(monkeypatch):
    """preflight_ort() must run before ensure_model/from_zipformer_ctc."""
    calls = []
    monkeypatch.setattr(asr, "preflight_ort", lambda: calls.append("preflight") or (1, 28, 2, 0))
    monkeypatch.setattr(
        asr, "ensure_model", lambda: calls.append("ensure_model") or ("m.onnx", "t.txt")
    )

    class _BoomRecognizer:
        @staticmethod
        def from_zipformer_ctc(**kwargs):
            calls.append("native")
            return object()

    monkeypatch.setattr(asr.sherpa_onnx.OfflineRecognizer, "from_zipformer_ctc",
                        _BoomRecognizer.from_zipformer_ctc, raising=False)
    asr.cache.clear()
    try:
        asr.load_align_model(device="cpu")
    finally:
        asr.cache.clear()
    assert calls == ["preflight", "ensure_model", "native"]


@pytest.mark.skipif(sys.platform != "win32", reason="Win32 version API")
def test_win_file_version_reads_real_dll():
    version = asr.bundled_ort_version()
    if version is None:
        pytest.skip("no bundled DLL in this env")
    major, minor, _, _ = version
    assert major == 1 and minor >= 20
