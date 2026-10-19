"""
sherpa-onnx CTC forced alignment.
Alternative to wav2vec2 alignment — uses zipformer-en-ctc.
"""

import logging, os, sys, numpy as np
from pathlib import Path

logger = logging.getLogger("WhisperCPP.AlignSherpa")

SHERPA_AVAILABLE = False
# Ganti ke Zipformer CTC yang lebih stabil
SHERPA_MODEL_NAME = "sherpa-onnx-zipformer-ctc-en-2023-10-02"

# Model dirs
try:
    import folder_paths
    MODEL_DIR = Path(folder_paths.models_dir) / "alignment" / "sherpa"
except ImportError:
    MODEL_DIR = Path(__file__).resolve().parent.parent.parent.parent / "models" / "alignment" / "sherpa"

MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    f"{SHERPA_MODEL_NAME}.tar.bz2"
)

try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
    logger.info("sherpa-onnx available")
except ImportError:
    logger.warning("sherpa-onnx not installed, run: pip install sherpa-onnx")


def ensure_model():
    """Download the Dolphin CTC model if needed."""
    model_dir = MODEL_DIR / SHERPA_MODEL_NAME
    model_file = model_dir / "model.int8.onnx"
    tokens_file = model_dir / "tokens.txt"
    if model_file.exists() and tokens_file.exists():
        return str(model_file), str(tokens_file)

    os.makedirs(MODEL_DIR, exist_ok=True)
    import urllib.request, tarfile
    archive = MODEL_DIR / f"{SHERPA_MODEL_NAME}.tar.bz2"
    logger.info(f"Downloading {SHERPA_MODEL_NAME} (81MB)...")
    urllib.request.urlretrieve(MODEL_URL, str(archive))
    with tarfile.open(str(archive)) as tar:
        tar.extractall(MODEL_DIR)
    archive.unlink()
    logger.info("Model downloaded and extracted")
    return str(model_file), str(tokens_file)

# Cache recognizer — auto-cleanup after 10 min inactive
# TODO: cleanup dipanggil dari whispercpp_node.py setelah transcribe
import time
cache = {}  # {device: (recognizer, last_used)}
CACHE_TTL = 600  # 10 menit

def _cleanup_stale():
    """Remove entries unused for >10 minutes."""
    global cache
    now = time.time()
    stale = [k for k, (_, ts) in cache.items() if now - ts > CACHE_TTL]
    for k in stale:
        del cache[k]
        logger.info(f"Alignment recognizer expired for {k}")

def cleanup():
    """Free all cached recognizers."""
    global cache
    cache.clear()
    logger.info("Alignment sherpa cache cleaned up")


# ── ONNX Runtime pairing guard (issue #17) ─────────────────────────────────
# sherpa-onnx's native module is compiled against a specific ONNX Runtime API.
# The process resolves `onnxruntime.dll` by NAME to whatever copy is already
# loaded (or to `sherpa_onnx/lib/` when none is); if that DLL is older than
# the API the module expects, the native GetApi() call returns NULL and the
# next use is a Windows access violation - uncatchable in Python, so ComfyUI
# itself dies and reconnects. Check the pairing in Python BEFORE entering
# native code and raise a normal exception instead; the node catches it and
# skips alignment.
from ..ort_util import dll_file_version, loaded_ort_modules, parse_so_version


class AlignmentUnavailableError(RuntimeError):
    """Alignment cannot run safely - raised instead of letting the process crash."""


# FileVersion of sherpa_onnx/lib/onnxruntime.dll shipped by each release
# (measured from the published win_amd64 wheels). Keep in sync when adding
# releases: a bundle older than this table means pyd/DLL skew -> native crash.
_KNOWN_BUNDLED_ORT = {
    "1.13.3": (1, 24, 4, 0),
    "1.13.4": (1, 27, 0, 0),
    "1.13.5": (1, 27, 1, 0),
    "1.13.6": (1, 27, 1, 0),
    "1.13.7": (1, 27, 1, 0),
    "1.13.8": (1, 28, 2, 0),
}


def bundled_ort_version():
    """Version of the onnxruntime bundled inside sherpa_onnx/lib, or None."""
    lib = Path(sherpa_onnx.__file__).resolve().parent / "lib"
    if sys.platform == "win32":
        dll = lib / "onnxruntime.dll"
        return dll_file_version(dll) if dll.exists() else None
    if lib.is_dir():
        for f in sorted(lib.iterdir()):
            v = parse_so_version(f.name)
            if v:
                return v
    return None


def preflight_ort():
    """Verify the sherpa/ONNX Runtime pairing before any native call (issue #17).

    Returns the bundled ORT version, or raises AlignmentUnavailableError with
    an actionable message. Without this check a skewed pairing kills the whole
    ComfyUI process with an uncatchable access violation.
    """
    version = getattr(sherpa_onnx, "__version__", "")
    expected = _KNOWN_BUNDLED_ORT.get(version)

    # The process may already contain an older onnxruntime.dll (e.g. a stale
    # build artifact pre-loaded at node import); by name-unification the
    # native module would bind to THAT copy and crash (issue #17).
    for path, ver in loaded_ort_modules():
        if expected and ver and ver[:2] < expected[:2]:
            raise AlignmentUnavailableError(
                f"an older onnxruntime.dll is already loaded in this process: "
                f"{path} ({ver[0]}.{ver[1]}.{ver[2]}) but sherpa-onnx {version} needs "
                f"{expected[0]}.{expected[1]}.x - continuing would crash the process "
                "(issue #17). Fix: update that DLL to >= the required version"
            )

    bundled = bundled_ort_version()
    if bundled is None:
        lib = Path(sherpa_onnx.__file__).resolve().parent / "lib"
        raise AlignmentUnavailableError(
            f"sherpa-onnx is missing its bundled onnxruntime.dll (expected in {lib}). "
            "Without it the native code loads an arbitrary older ONNX Runtime and "
            "crashes the process (issue #17). Fix: pip install --force-reinstall sherpa-onnx-core"
        )
    if expected and bundled[:2] < expected[:2]:
        raise AlignmentUnavailableError(
            f"sherpa-onnx {version} has a mismatched ONNX Runtime bundle: "
            f"onnxruntime.dll is {bundled[0]}.{bundled[1]}.{bundled[2]} but its native "
            f"module needs {expected[0]}.{expected[1]}.x - loading the model would crash "
            "the process (issue #17). Fix: "
            f"pip install --force-reinstall sherpa-onnx=={version} sherpa-onnx-core=={version}"
        )
    logger.info(f"  ORT pairing OK: sherpa-onnx {version} + onnxruntime "
                f"{bundled[0]}.{bundled[1]}.{bundled[2]}")
    return bundled


def load_align_model(device="cpu"):
    """Load sherpa-onnx CTC recognizer. Cached with 10min TTL."""
    global cache
    if not SHERPA_AVAILABLE:
        raise ImportError("sherpa-onnx not installed")

    # Clean expired entries
    _cleanup_stale()

    # Return cached if available
    if device in cache:
        recognizer, _ = cache[device]
        cache[device] = (recognizer, time.time())  # refresh TTL
        logger.debug(f"Alignment recognizer cached for {device}")
        return recognizer

    # Never enter the native ORT code with a mismatched runtime (issue #17):
    # a NULL OrtApi pointer there is an uncatchable access violation.
    preflight_ort()

    model_path, tokens_path = ensure_model()
    logger.info(f"Loading sherpa CTC model: {model_path}")
    # sherpa-onnx hanya support cpu & cuda provider. Map device dropdown:
    #   cuda       -> cuda (NVIDIA)
    #   auto       -> cuda kalau NVIDIA terdeteksi, else cpu
    #   vulkan/opencl/metal/hip/cpu -> cpu (alignment ringan, GPU lain gak diuntungkan)
    if device == "cuda":
        provider = "cuda"
    elif device == "auto":
        try:
            from ..gpu_detect import detect_gpu
            provider = "cuda" if detect_gpu()["backend"] == "cuda" else "cpu"
        except Exception:
            provider = "cpu"
    else:
        provider = "cpu"
    logger.info(f"  sherpa provider: {provider}")
    recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
        model=model_path,
        tokens=tokens_path,
        provider=provider,
        debug=False,
    )
    cache[device] = (recognizer, time.time())
    return recognizer


def _decode_tokens(tokens, token_ids):
    """Decode BPE token IDs back to text (basic SentencePiece cleanup)."""
    # tokens_file format: each line is a token string (0-indexed)
    text_parts = []
    for tid in token_ids:
        if 0 <= tid < len(tokens):
            tok = tokens[tid]
            # SentencePiece format: space-prefixed tokens start words
            if tok.startswith("_"):
                text_parts.append(" " + tok[1:])
            elif tok.startswith("▁"):
                text_parts.append(" " + tok[1:])
            else:
                text_parts.append(tok)
    return "".join(text_parts).strip()


def _merge_tokens_to_words(tokens_list, timestamps_list):
    """
    Merge CTC subword tokens into words with start/end timestamps.
    
    Returns: list of {"word": str, "start": float, "end": float}
    """
    if not tokens_list or not timestamps_list:
        return []
    
    words = []
    current_word_parts = []
    current_start = timestamps_list[0]
    
    for tok, ts in zip(tokens_list, timestamps_list):
        # If this token starts a new word (SentencePiece convention)
        # Space/underscore prefix = word start
        is_word_start = tok.startswith("_") or tok.startswith("▁") or tok.startswith(" ")
        
        if is_word_start and current_word_parts:
            # Flush previous word
            word_text = "".join(current_word_parts).strip()
            if word_text:
                words.append({
                    "word": word_text,
                    "start": current_start,
                    "end": ts,
                })
            current_word_parts = [tok.lstrip("_▁ ")]
            current_start = ts
        else:
            current_word_parts.append(tok)
    
    # Flush last word
    if current_word_parts:
        word_text = "".join(current_word_parts).strip()
        if word_text:
            words.append({
                "word": word_text,
                "start": current_start,
                "end": timestamps_list[-1] + 0.1,  # estimate end
            })
    
    return words


def align(segments, audio_data, recognizer, language="en"):
    """
    Align whisper segments using sherpa-onnx CTC model.
    
    Args:
        segments: list of whisper segment dicts with 'text' and optional 'words'
        audio_data: numpy array (float32, mono) of full audio
        recognizer: sherpa_onnx.OfflineRecognizer (CTC)
        language: language code (not used for CTC, kept for API compat)
    
    Returns:
        Updated segments with word-level timestamps.
    """
    sr = 16000  # CTC model expects 16kHz
    
    # Resample audio if needed
    if audio_data is None or len(audio_data) == 0:
        return segments
    
    # Chunk audio by whisper segments to avoid OOM on long audio
    # Process each segment independently through CTC model
    total_samples = len(audio_data)
    logger.info(f"Sherpa CTC alignment: {len(segments)} segments, {total_samples/sr:.1f}s audio")
    
    all_aligned_words = []
    segment_word_mapping = {}  # whisper word text -> (start, end)
    
    for seg_idx, seg in enumerate(segments):
        start_sample = int(seg.get("start", 0) * sr)
        end_sample = int(seg.get("end", total_samples / sr) * sr)
        end_sample = min(end_sample, total_samples)
        
        # Add 0.5s context on each side for CTC alignment accuracy
        ctx_start = max(0, start_sample - int(0.5 * sr))
        ctx_end = min(total_samples, end_sample + int(0.5 * sr))
        seg_pad_start = start_sample - ctx_start  # padding in samples before seg
        
        # Skip very short segments (CTC needs minimum audio)
        if ctx_end - ctx_start < sr * 0.3:  # Less than 300ms
            continue
        
        chunk = audio_data[ctx_start:ctx_end]
        try:
            stream = recognizer.create_stream()
            stream.accept_waveform(sr, chunk.tolist())
            recognizer.decode_stream(stream)
            result = stream.result
            if not result or not result.tokens:
                continue
            words = _merge_tokens_to_words(result.tokens, result.timestamps)
            # Adjust timestamps back to full audio reference
            for w in words:
                w["start"] = round(w["start"] + ctx_start / sr, 3)
                w["end"] = round(w["end"] + ctx_start / sr, 3)
            all_aligned_words.extend(words)
            logger.debug(f"  Seg {seg_idx}: {len(words)} words from CTC")
        except Exception as e:
            logger.warning(f"  Seg {seg_idx} CTC failed: {e}")
            continue
    
    if not all_aligned_words:
        logger.warning("Sherpa CTC produced no tokens for any segment")
        return segments
    
    logger.info(f"CTC alignment: {len(all_aligned_words)} words total from sherpa")
    
    # Build word list from whisper segments
    all_whisper_words = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            for w in seg_words:
                w["segment_index"] = len(all_whisper_words)
                all_whisper_words.append(w)
        else:
            for w in seg.get("text", "").split():
                all_whisper_words.append({
                    "word": w,
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "segment_index": len(all_whisper_words),
                })
    
    if not all_whisper_words:
        return segments
    
    # Simple sequential word mapping
    from difflib import SequenceMatcher
    sherpa_strs = [w["word"].lower() for w in all_aligned_words]
    whisper_strs = [w["word"].lower() for w in all_whisper_words]
    matcher = SequenceMatcher(None, sherpa_strs, whisper_strs)
    mapping = {}
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                mapping[j1 + offset] = (i1 + offset, 1.0)
        elif tag == "replace":
            min_len = min(i2 - i1, j2 - j1)
            for offset in range(min_len):
                mapping[j1 + offset] = (i1 + offset, 0.7)
    
    # Apply timestamps
    for w_idx, w in enumerate(all_whisper_words):
        if w_idx in mapping:
            s_idx, _ = mapping[w_idx]
            if s_idx < len(all_aligned_words):
                w["start"] = round(all_aligned_words[s_idx]["start"], 3)
                w["end"] = round(all_aligned_words[s_idx]["end"], 3)
    
    # Rebuild segment word lists
    for seg in segments:
        seg_words = seg.get("words", [])
        for sw in seg_words:
            for aw in all_whisper_words:
                if (aw.get("word") == sw.get("word") and
                    aw.get("segment_index") == sw.get("segment_index", 0) and
                    "start" in aw and "end" in aw):
                    sw["start"] = aw["start"]
                    sw["end"] = aw["end"]
                    break
        if seg_words:
            seg["start"] = seg_words[0].get("start", seg.get("start", 0))
            seg["end"] = seg_words[-1].get("end", seg.get("end", 0))
    
    return segments
