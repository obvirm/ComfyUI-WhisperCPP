import ctypes, logging, os, platform
from pathlib import Path

from .ort_util import dll_file_version, pick_newest_ort, sherpa_bundle_dll

# Pre-load onnxruntime.dll BEFORE any other module loads a different version.
# The process can only use ONE onnxruntime.dll - the loader resolves that name
# to whatever is already loaded - so the first copy in wins for every later
# dependent (cpp-annote AND sherpa-onnx; issue #17). Pre-load the NEWEST copy
# available: a stale build artifact here used to poison the process and crash
# ComfyUI with an uncatchable access violation whenever align=true
# (sherpa-onnx needs ORT API 28, pre-loaded 1.27.0 -> GetApi returns NULL).
_NODE_DIR = Path(__file__).resolve().parent.parent
_LOG = logging.getLogger("WhisperCPP")


def _ort_candidates():
    cands = [
        _NODE_DIR / "onnxruntime.dll",
        _NODE_DIR / "cpp-annote" / "build" / "Release" / "onnxruntime.dll",
        sherpa_bundle_dll(),
    ]
    return cands


_picked = pick_newest_ort(_ort_candidates())
if _picked is not None:
    try:
        ctypes.CDLL(str(_picked))
        _LOG.info(f"Pre-loaded onnxruntime: {_picked} (v{dll_file_version(_picked)})")
        if platform.system() == "Windows":
            os.add_dll_directory(str(_picked.parent.resolve()))
    except Exception as _e:
        _LOG.warning(f"onnxruntime pre-load failed: {_e}")

from .whisper_lib import WhisperCPP
from .audio import AudioProcessor

__all__ = ["WhisperCPP", "AudioProcessor"]
