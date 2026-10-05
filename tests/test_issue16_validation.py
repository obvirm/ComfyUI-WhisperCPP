"""Cross-check issue #16 against ComfyUI's real prompt validator.

The bug report was: selecting ``日本語 (Japanese)`` or ``Multilingual / mixed``
from the WhisperCPP language dropdown rejected the prompt with::

    Value not in list: language: '日本語 (Japanese)' not in (list of length 203)

So the fix has to be proven against ``execution.validate_inputs`` - the exact
function ComfyUI runs before a node executes - not against a re-implementation
of its rules.

Skipped automatically when a ComfyUI checkout (nodes.py) is not the parent of
``custom_nodes/``, i.e. on bare CI runners.
"""
import asyncio
import importlib
import inspect
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]  # <ComfyUI>/custom_nodes/<this repo>/tests -> <ComfyUI>

if not (COMFY_ROOT / "nodes.py").exists():
    pytest.skip("ComfyUI checkout not found next to custom_nodes/", allow_module_level=True)

if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))

pytest.importorskip("torch")
execution = pytest.importorskip("execution")
nodes = pytest.importorskip("nodes")

# Import the node module with package context so its relative imports work,
# without running the repo-root __init__.py (that file is the ComfyUI loader).
_PKG = "whisperxx_issue16_under_test"
if _PKG not in sys.modules:
    _pkg = types.ModuleType(_PKG)
    _pkg.__path__ = [str(REPO_ROOT)]
    sys.modules[_PKG] = _pkg
node_mod = importlib.import_module(f"{_PKG}.whispercpp_node")
WhisperCPPNode = node_mod.WhisperCPPNode

# Register only for this validator run; never touch a live server mapping.
if "WhisperCPPNode" not in nodes.NODE_CLASS_MAPPINGS:
    nodes.NODE_CLASS_MAPPINGS["WhisperCPPNode"] = WhisperCPPNode

# Labels reported in issue #16 (injected into the dropdown by MiniMax Music
# Production Toolkit, and stored in saved workflows).
FOREIGN_LANGUAGE_VALUES = ["日本語 (Japanese)", "Multilingual / mixed"]
VALID_LANGUAGE_VALUES = FOREIGN_LANGUAGE_VALUES + ["None", "auto", "ja", "Japanese", "en"]
INVALID_LANGUAGE_VALUES = ["Klingon", "日本語", "日本語 (Klingon)"]


def _validate(language, **overrides):
    prompt = {
        "1": {
            "class_type": "WhisperCPPNode",
            "inputs": {
                "audio": {"waveform": [[[]]], "sample_rate": 16000},
                "model": "large-v3-turbo",
                "language": language,
                "task": "transcribe",
                "n_threads": 4,
                "device": "auto",
                **overrides,
            },
        }
    }
    valid, errors, _uid = asyncio.run(execution.validate_inputs("issue16", prompt, "1", {}))
    return valid, errors


def _error_types(errors):
    return {e.get("type") for e in errors}


def test_declares_language_for_custom_validation():
    """ComfyUI only skips the combo check when `language` is an argument."""
    args = inspect.getfullargspec(WhisperCPPNode.VALIDATE_INPUTS).args
    assert "language" in args, "VALIDATE_INPUTS must take `language` to skip the list check"


@pytest.mark.parametrize("language", VALID_LANGUAGE_VALUES)
def test_language_values_pass_prompt_validation(language):
    valid, errors = _validate(language)
    assert valid, f"language={language!r} rejected: {errors}"
    assert errors == []


@pytest.mark.parametrize("language", INVALID_LANGUAGE_VALUES)
def test_unmapped_language_still_accepted_then_normalized(language):
    """Unknown selectors must not kill the prompt either - the node falls back
    to auto-detect and logs a warning (issue #16: never block the workflow)."""
    valid, errors = _validate(language)
    assert valid, f"language={language!r} rejected: {errors}"


def test_other_inputs_are_still_validated():
    """Skipping the check is scoped to `language` only."""
    valid, errors = _validate("en", model="definitely-not-a-model")
    assert not valid
    assert "value_not_in_list" in _error_types(errors)

    valid, errors = _validate("en", n_threads="not-an-int")
    assert not valid
    assert _error_types(errors) & {"invalid_input_type", "value_not_in_list"}


def test_non_string_language_is_rejected():
    valid, errors = _validate(12345)
    assert not valid
    assert "custom_validation_failed" in _error_types(errors)


def test_input_types_language_list_is_clean_and_complete():
    lang_options = WhisperCPPNode.INPUT_TYPES()["required"]["language"][0]
    assert lang_options[:2] == ["None", "auto"]
    assert {"en", "ja", "Japanese", "None", "auto"} <= set(lang_options)
    for label in FOREIGN_LANGUAGE_VALUES:
        assert label not in lang_options
    assert len(lang_options) == len(set(lang_options))


# ── Node plumbing: what actually reaches whisper.cpp ────────────────────────


class _DummyMgr:
    def get_model_path(self, key):
        return "dummy-model.ggml"


class _DummyWhisper:
    _lib = None
    _ctx = None

    def __init__(self):
        self.last_kwargs = None

    def load_model(self, *args, **kwargs):
        return None

    def transcribe(self, audio, **kwargs):
        self.last_kwargs = kwargs
        return {
            "text": "hello",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": kwargs.get("language") or "en",
            "n_segments": 1,
        }


@pytest.fixture
def dummy_node(monkeypatch):
    dummy = _DummyWhisper()
    monkeypatch.setattr(WhisperCPPNode, "_whisper", dummy, raising=False)
    monkeypatch.setattr(WhisperCPPNode, "_model_manager", _DummyMgr(), raising=False)
    return WhisperCPPNode(), dummy


def _run(node, language):
    return node.transcribe(
        audio={"waveform": [0.0] * 16000, "sample_rate": 16000},
        model="large-v3-turbo",
        language=language,
        task="transcribe",
        n_threads=2,
        device="cpu",
        align=False,          # no sherpa model download in tests
        hallu_filter=False,   # go straight to wcpp.transcribe
        vad=False,
        diarize=False,
        separate_vocals=False,
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("日本語 (Japanese)", "ja"),
        ("Multilingual / mixed", None),
        ("None", None),
        ("auto", None),
        ("Japanese", "ja"),
        ("ja", "ja"),
        ("Klingon", None),  # unknown -> auto-detect + warning, never a crash
    ],
)
def test_transcribe_receives_normalized_language(dummy_node, raw, expected):
    node, dummy = dummy_node
    outputs = _run(node, raw)
    assert dummy.last_kwargs is not None, "whisper.transcribe was never called"
    assert dummy.last_kwargs["language"] == expected
    assert dummy.last_kwargs["detect_language"] is (expected is None)
    assert outputs[0] == "hello", outputs[0]
