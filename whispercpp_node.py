import gc, io, json, logging, os, sys, time
from typing import Optional
import comfy.utils, comfy.model_management, folder_paths
import numpy as np
import torch, torchaudio

from .whispercpp.whisper_lib import WhisperCPP
from .whispercpp.audio import AudioProcessor, WHISPER_SAMPLE_RATE

# Model Manager — wrapped agar node tetap register meski tqdm/requests belum terinstall
WHISPERCPP_MODEL_AVAILABLE = False
try:
    from .whispercpp.model import ModelManager, get_model_keys, load_custom_models
    WHISPERCPP_MODEL_AVAILABLE = True
except ImportError:
    # Fallback: node tetap jalan, tapi download model otomatis nggak akan work
    class DummyModelManager:
        def __init__(self): self._model_path = None
        def ensure_custom_config(self, *args): pass
        def get_model_path(self, key): return None
        def download_model(self, key):
            logger = logging.getLogger("WhisperCPP")
            logger.error(f"Cannot download {key}: install tqdm & requests: pip install tqdm requests huggingface-hub")
            return None
    ModelManager = DummyModelManager
    GGML_FALLBACK_KEYS = ["large-v3-turbo","large-v3","medium","small","base","tiny","tiny.en","base.en","small.en","medium.en","large-v2"]
    def get_model_keys(): return GGML_FALLBACK_KEYS
    def load_custom_models(*args, **kwargs): pass

# Language codes / names / selector normalization live in whispercpp.languages
# (issue #16) so they can be unit-tested without the ComfyUI runtime.
from .whispercpp.languages import (
    WHISPER_LANGUAGES,
    LANGUAGES,
    TO_LANGUAGE_CODE,
    language_options,
    resolve_language,
)


def shift_segment_timestamps(segment: dict, offset: float) -> dict:
    """Shift a parent segment AND its nested words to absolute full-audio time.

    When audio is split into chunks (VAD / RMS hallucination filter) the node
    rewinds each chunk to local time. The parent segment already gets shifted by
    `offset`, but the nested word timestamps must receive the same offset or
    they jump backward at every new chunk boundary.
    """
    segment["start"] = segment.get("start", 0) + offset
    segment["end"] = segment.get("end", 0) + offset
    for word in segment.get("words", []):
        local_start = word.get("start", 0)
        local_end = word.get("end", local_start)
        word["start"] = local_start + offset
        word["end"] = local_end + offset
    return segment

# Standalone alignment — sherpa-onnx CTC (no wav2vec2, no whisperx)
SHERPA_ALIGN_AVAILABLE = False
try:
    from .whispercpp.ext.alignment_sherpa import (
        load_align_model, align, SHERPA_AVAILABLE
    )
    SHERPA_ALIGN_AVAILABLE = SHERPA_AVAILABLE
except ImportError:
    pass

NODE_DIR = os.path.dirname(os.path.abspath(__file__))
GGML_MODEL_KEYS = get_model_keys()

class ColoredLogger:
    def __init__(self, name="WhisperCPPNode"):
        self.name = name
        self.C = {"reset":"\033[0m","info":"\033[94m","success":"\033[92m","warning":"\033[93m","error":"\033[91m"}
    def _log(self, m, l): print(f"{self.C.get(l,self.C['reset'])}[{self.name}] {m}{self.C['reset']}")
    def info(self, m): self._log(m, "info")
    def success(self, m): self._log(m, "success")
    def warning(self, m): self._log(m, "warning")
    def error(self, m): self._log(m, "error")

logger = ColoredLogger()

# Standalone UVR vocal separation
UVR_AVAILABLE = False
try:
    from .whispercpp.ext.uvr import separate_vocals
    UVR_AVAILABLE = True
    logger.info("UVR vocal separation available")
except Exception as e:
    logger.debug(f"UVR import: {e}")

# cpp-annote VAD/Diarization (DLL)
CPPANNOTE_AVAILABLE = False
try:
    from .whispercpp.ext.cppannote import segment as _vad_seg, CPPANNOTE_AVAILABLE as _ca
    CPPANNOTE_AVAILABLE = _ca
    if CPPANNOTE_AVAILABLE:
        logger.info("cpp-annote VAD available")
except Exception as e:
    logger.debug(f"cpp-annote import: {e}")

class WhisperCPPNode:
    PBAR_FORMAT = "\033[96m{l_bar}\033[0m\033[92m{bar:15}\033[0m\033[93m{r_bar}\033[0m"
    _whisper = None
    _model_manager = None

    @classmethod
    def INPUT_TYPES(cls):
        lang_list = language_options()

        required = {
            "audio": ("AUDIO",),
            "model": (GGML_MODEL_KEYS, {"default": "large-v3-turbo"}),
            "language": (lang_list, {"default": "en"}),
            "task": (["transcribe", "translate"],),
            "n_threads": ("INT", {"default": 4, "min": 1, "max": 64}),
            "device": (["auto", "cuda", "cpu", "vulkan", "metal", "opencl", "hip"], {"default": "auto"}),
        }
        # --- SELALU KELIHATAN (umum / plug-and-play) ---
        optional = {
            "separate_vocals": ("BOOLEAN", {"default": False}),
            "dtw_token_timestamps": ("BOOLEAN", {"default": False}),
            "vad": ("BOOLEAN", {"default": False}),
            "align": ("BOOLEAN", {"default":True}),

            # --- ADVANCE CPP (show_advance_cpp) ---
            "show_advance_cpp": ("BOOLEAN", {"default": False}),
            "sampling_strategy": (["greedy","beam_search"], {"default":"greedy"}),
            "best_of": ("INT", {"default":5,"min":1,"max":20}),
            "beam_size": ("INT", {"default":5,"min":1,"max":20}),
            "patience": ("FLOAT", {"default":-1.0,"min":0.0,"max":10.0,"step":0.1}),
            "temperature": ("FLOAT", {"default":0.0,"min":0.0,"max":2.0,"step":0.1}),
            "temperature_inc": ("FLOAT", {"default":0.4,"min":0.0,"max":1.0,"step":0.1}),
            "max_initial_ts": ("FLOAT", {"default":1.0,"min":0.0,"max":60.0,"step":0.1}),
            "length_penalty": ("FLOAT", {"default":-1.0,"min":-10.0,"max":5.0,"step":0.1}),
            "n_max_text_ctx": ("INT", {"default":16384,"min":-1,"max":65536}),
            "offset_ms": ("INT", {"default":0,"min":0,"max":3600000}),
            "duration_ms": ("INT", {"default":0,"min":0,"max":3600000}),
            "no_context": ("BOOLEAN", {"default":True}),
            "single_segment": ("BOOLEAN", {"default":False}),
            "no_timestamps": ("BOOLEAN", {"default":False}),
            "max_tokens": ("INT", {"default":0,"min":0,"max":1024}),
            "max_len": ("INT", {"default":0,"min":0,"max":512}),
            "split_on_word": ("BOOLEAN", {"default":False}),
            "token_timestamps": ("BOOLEAN", {"default":False}),
            "thold_pt": ("FLOAT", {"default":0.01,"min":0.0,"max":1.0,"step":0.001}),
            "thold_ptsum": ("FLOAT", {"default":0.01,"min":0.0,"max":1.0,"step":0.001}),
            "suppress_blank": ("BOOLEAN", {"default":True}),
            "suppress_nst": ("BOOLEAN", {"default":True}),
            "hallu_filter": ("BOOLEAN", {"default":True}),
            "hallu_threshold": ("FLOAT", {"default":0.6,"min":0.0,"max":1.0,"step":0.05}),
            "suppress_regex": ("STRING", {"default":""}),
            "entropy_thold": ("FLOAT", {"default":2.0,"min":0.0,"max":10.0,"step":0.1}),
            "logprob_thold": ("FLOAT", {"default":-0.5,"min":-10.0,"max":0.0,"step":0.1}),
            "no_speech_thold": ("FLOAT", {"default":0.6,"min":0.0,"max":1.0,"step":0.01}),
            "initial_prompt": ("STRING", {"default":"","multiline":True}),
            "carry_initial_prompt": ("BOOLEAN", {"default":False}),
            "audio_ctx": ("INT", {"default":0,"min":0,"max":4096}),
            "debug_mode": ("BOOLEAN", {"default":False}),
            "print_special": ("BOOLEAN", {"default":False}),
            "print_progress": ("BOOLEAN", {"default":True}),
            "tdrz_enable": ("BOOLEAN", {"default":False}),
            "flash_attn": ("BOOLEAN", {"default":False}),
            "gpu_device": ("INT", {"default":0,"min":-1,"max":8}),
            "dtw_aheads_preset": (["none","n_top_most","custom","tiny_en","tiny","base_en","base","small_en","small","medium_en","medium","large_v1","large_v2","large_v3","large_v3_turbo"], {"default":"large_v3_turbo"}),
            "dtw_n_top": ("INT", {"default":-1,"min":-1,"max":64}),
            "grammar_penalty": ("FLOAT", {"default":100.0,"min":0.0,"max":100.0,"step":0.1}),

            # --- ADVANCE EXT (show_advance_ext): UVR + alignment + diarization ---
            "show_advance_ext": ("BOOLEAN", {"default": False}),
            "separate_model": (["voc_fv6-Q8_0.gguf","BSRoformer-anvuew-Q8_0.gguf","becruily_deux-Q8_0.gguf","voc_fv6-FP16.gguf"], {"default":"voc_fv6-Q8_0.gguf"}),
            "separate_chunk_size": ("INT", {"default":-1,"min":-1,"max":1000000,"step":1}),
            "separate_overlap": ("INT", {"default":-1,"min":-1,"max":20,"step":1}),
            "align_model": (["sherpa-onnx-zipformer-ctc-en-2023-10-02"], {"default":"sherpa-onnx-zipformer-ctc-en-2023-10-02"}),
            "return_char_alignments": ("BOOLEAN", {"default":False}),
            "diarize": ("BOOLEAN", {"default":False}),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("STRING","STRING","STRING","STRING","STRING","STRING","STRING")
    RETURN_NAMES = ("text","segments_json","srt","vtt","tsv","aud","json_result")
    FUNCTION = "transcribe"
    CATEGORY = "WhisperCPP"

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, language):
        """Accept every language selector ComfyUI might hand us (issue #16).

        Declaring ``language`` here makes ComfyUI skip its default
        ``value not in list`` check for this input ONLY (see execution.py,
        ``if x not in validate_function_inputs``); every other input keeps the
        usual validation.

        Needed because other extensions inject labels such as
        ``日本語 (Japanese)`` / ``Multilingual / mixed`` into the dropdown, and
        those values are stored in saved workflows. They are normalized to
        whisper.cpp codes (or auto-detect) by ``resolve_language`` at run time
        instead of killing the prompt before transcription starts.
        """
        if language is None:
            return True
        if not isinstance(language, str):
            return f"language must be a string, got {type(language).__name__}"
        return True

    def _get(self, kw, key, default=None):
        v = kw.get(key); return v if v is not None else default

    def _ensure_whisper(self):
        if WhisperCPPNode._whisper is None: WhisperCPPNode._whisper = WhisperCPP()
        return WhisperCPPNode._whisper

    def _ensure_mgr(self):
        if WhisperCPPNode._model_manager is None:
            WhisperCPPNode._model_manager = ModelManager()
        return WhisperCPPNode._model_manager

    def transcribe(self, **kwargs):
        logger.info("="*60); logger.info("  WhisperCPP Node starting"); logger.info("="*60)
        logger.info("Step 1: Creating ProgressBar")
        pbar = comfy.utils.ProgressBar(7)
        logger.info("Step 2: Creating cache directory")

        hf_cache = os.path.join(folder_paths.models_dir, "whispercpp")
        os.makedirs(hf_cache, exist_ok=True)
        pbar.update(1)
        logger.info("Step 3: Getting model key")

        model_key = self._get(kwargs,"model","large-v3-turbo")
        logger.info(f"Step 4: Model key = {model_key}")
        mgr = self._ensure_mgr()
        logger.info("Step 5: Getting model path")
        model_path = mgr.get_model_path(model_key)
        logger.info(f"Step 6: Model path = {model_path}")
        if not model_path:
            logger.info(f"Downloading {model_key}...")
            model_path = mgr.download_model(model_key)
            if not model_path: logger.error("Model failed"); return ("","","","","","","")
        pbar.update(1)

        device = self._get(kwargs,"device","auto")
        use_gpu = device in ("auto","cuda","vulkan","metal","hip")
        gpu_device = self._get(kwargs,"gpu_device",0)

        # Map device to backend for each module.
        # Prioritas backend proper (FULL): CUDA > Vulkan > OpenCL > Metal > CPU.
        # 'auto' = biarkan whisper/bsr pilih device GPU terbaik yang tersedia
        # (ggml enumerasi CUDA duluan, lalu Vulkan/OpenCL/Metal sebagai fallback).
        def _device_to_backend(dev):
            """Map device dropdown to backend name."""
            if dev in ("auto", "cuda"): return "cuda"
            if dev == "vulkan": return "vulkan"
            if dev == "metal": return "metal"
            if dev == "opencl": return "opencl"
            if dev == "hip": return "rocm"
            if dev == "cpu": return "cpu"
            return "auto"
        backend = _device_to_backend(device)
        
        wcpp = self._ensure_whisper()
        dtw_preset_str = self._get(kwargs,"dtw_aheads_preset","large_v3_turbo")
        if not isinstance(dtw_preset_str, str) or dtw_preset_str not in {"none","n_top_most","custom","tiny_en","tiny","base_en","base","small_en","small","medium_en","medium","large_v1","large_v2","large_v3","large_v3_turbo"}:
            dtw_preset_str = "large_v3_turbo"
        dtw_preset = {"none":0,"n_top_most":1,"custom":2,"tiny_en":3,"tiny":4,"base_en":5,"base":6,"small_en":7,"small":8,"medium_en":9,"medium":10,"large_v1":11,"large_v2":12,"large_v3":13,"large_v3_turbo":14}[dtw_preset_str]
        logger.info("Step 10: Loading whisper model (may take time)...")
        try: wcpp.load_model(model_path, use_gpu=use_gpu, gpu_device=gpu_device, flash_attn=self._get(kwargs,"flash_attn",False),
            dtw_token_timestamps=self._get(kwargs,"dtw_token_timestamps",False), dtw_aheads_preset=dtw_preset, dtw_n_top=self._get(kwargs,"dtw_n_top",-1))
        except RuntimeError as e: logger.error(f"Load failed: {e}"); return ("","","","","","","")
        logger.info("Step 11: Model loaded successfully")
        pbar.update(1)

        audio_data = AudioProcessor.process_comfy_audio(kwargs.get("audio"))
        
        # Validate audio input
        if audio_data is None or len(audio_data) == 0:
            logger.error("Audio input is empty")
            return ("", "", "", "", "", "", "")
        if len(audio_data) < 1600:  # Less than 0.1s at 16kHz
            logger.error(f"Audio too short: {len(audio_data)} samples ({len(audio_data)/16000:.2f}s)")
            return ("", "", "", "", "", "", "")
        
        # --- Optional UVR vocal separation (BSRoformer) ---
        if self._get(kwargs,"separate_vocals",False):
            separate_model = self._get(kwargs,"separate_model","voc_fv6-Q8_0.gguf")
            separate_chunk_size = self._get(kwargs,"separate_chunk_size",-1)
            separate_overlap = self._get(kwargs,"separate_overlap",-1)
            try:
                # Audio is 16kHz from process_comfy_audio; resample to 44.1kHz for BSRoformer
                t = torch.from_numpy(audio_data).float().unsqueeze(0)
                audio_44k = torchaudio.transforms.Resample(WHISPER_SAMPLE_RATE, 44100)(t).squeeze().numpy().astype(np.float32)
                del t  # Free torch tensor
                audio_44k = separate_vocals(audio_44k, sr=44100,
                    model_name=separate_model, chunk_size=separate_chunk_size, overlap=separate_overlap, use_gpu=use_gpu)
                # UVR outputs 44.1kHz; resample back to 16kHz for whisper
                t2 = torch.from_numpy(audio_44k).float().unsqueeze(0)
                del audio_44k  # Free intermediate array
                audio_data = torchaudio.transforms.Resample(44100, WHISPER_SAMPLE_RATE)(t2).squeeze().numpy().astype(np.float32)
                del t2  # Free torch tensor
                logger.info(f"UVR done: {len(audio_data)} samples")
            except Exception as e:
                logger.error(f"UVR failed: {e}")
        pbar.update(1)

        # Issue #16: the value may be a code, an English name, a foreign label
        # ("日本語 (Japanese)") or an auto synonym ("None", "auto",
        # "Multilingual / mixed"). Normalize instead of passing junk to
        # whisper.cpp (which errors out on unknown languages).
        lang_raw = self._get(kwargs, "language", "en")
        lang, lang_known = resolve_language(lang_raw)
        if not lang_known:
            logger.warning(f"Unknown language selector {lang_raw!r} -> auto-detect")
        logger.info(f"Language: {lang or 'auto-detect'} (raw={lang_raw!r})")
        strat_str = self._get(kwargs,"sampling_strategy","greedy")
        strategy = 0 if strat_str == "greedy" else 1

        # Common transcribe params (nggak include offset/duration biar bisa per-segment)
        tp_base = { "strategy":strategy, "n_threads":self._get(kwargs,"n_threads",4), "language":lang, "detect_language":bool(lang is None), "task":self._get(kwargs,"task","transcribe"),
            "temperature":self._get(kwargs,"temperature",0.0), "temperature_inc":self._get(kwargs,"temperature_inc",0.4), "max_initial_ts":self._get(kwargs,"max_initial_ts",1.0), "length_penalty":self._get(kwargs,"length_penalty",-1.0),
            "best_of":self._get(kwargs,"best_of",5), "beam_size":self._get(kwargs,"beam_size",5), "patience":self._get(kwargs,"patience",-1.0),
            "entropy_thold":self._get(kwargs,"entropy_thold",2.0), "logprob_thold":self._get(kwargs,"logprob_thold",-0.5), "no_speech_thold":self._get(kwargs,"no_speech_thold",0.6),
            "n_max_text_ctx":self._get(kwargs,"n_max_text_ctx",16384), "no_context":self._get(kwargs,"no_context",True),
            "no_timestamps":self._get(kwargs,"no_timestamps",False),
            "max_tokens":self._get(kwargs,"max_tokens",0), "max_len":self._get(kwargs,"max_len",0), "split_on_word":self._get(kwargs,"split_on_word",False),
            "token_timestamps":self._get(kwargs,"token_timestamps",False), "thold_pt":self._get(kwargs,"thold_pt",0.01), "thold_ptsum":self._get(kwargs,"thold_ptsum",0.01),
            "suppress_blank":self._get(kwargs,"suppress_blank",True), "suppress_nst":self._get(kwargs,"suppress_nst",True), "suppress_regex":self._get(kwargs,"suppress_regex","") or None,
            "initial_prompt":self._get(kwargs,"initial_prompt","") or None, "carry_initial_prompt":self._get(kwargs,"carry_initial_prompt",False),
            "audio_ctx":self._get(kwargs,"audio_ctx",0), "debug_mode":self._get(kwargs,"debug_mode",False),
            "print_special":self._get(kwargs,"print_special",False), "print_progress":self._get(kwargs,"print_progress",True),
            "tdrz_enable":self._get(kwargs,"tdrz_enable",False),
            "grammar_penalty":self._get(kwargs,"grammar_penalty",100.0), "vad":False }

        # ── VAD: cpp-annote (DLL) untuk speech segmentation ──
        do_vad = self._get(kwargs,"vad",False)

        if do_vad and CPPANNOTE_AVAILABLE:
            try:
                from .whispercpp.ext.cppannote import segment as vad_segment
                speech_segs = vad_segment(audio_data, sr=16000)
                logger.info(f"VAD: {len(speech_segs)} speech segment(s)")
                for s, e in speech_segs:
                    logger.info(f"  {s:.2f}s -> {e:.2f}s")
            except Exception as e:
                logger.warning(f"VAD failed, fallback full audio: {e}")
                speech_segs = [(0.0, len(audio_data)/16000)]
                do_vad = False
        elif do_vad:
            logger.warning("cpp-annote not available, VAD disabled")
            do_vad = False

        # ── Transcribe ──
        if do_vad:
            # Per-segment transcription
            all_segments = []
            full_texts = []
            for seg_i, (seg_start, seg_end) in enumerate(speech_segs):
                comfy.model_management.throw_exception_if_processing_interrupted()
                dur_ms = int((seg_end - seg_start) * 1000)
                if dur_ms < 100:  # skip <100ms segments
                    continue
                # Extrak chunk — NO initial_prompt (causes whisper loops)
                chunk = audio_data[int(seg_start*16000):int(seg_end*16000)]
                if len(chunk) == 0:
                    continue
                seg_tp = {**tp_base, "offset_ms": 0, "duration_ms": 0, "initial_prompt": None}
                try:
                    seg_result = wcpp.transcribe(chunk, **seg_tp)
                    seg_text = seg_result.get("text","").strip()
                    if seg_text:
                        full_texts.append(seg_text)
                    for seg in seg_result.get("segments", []):
                        # Shift chunk timestamps (segment + nested words) to
                        # original audio position.
                        seg = shift_segment_timestamps(seg, seg_start)
                        all_segments.append(seg)
                    logger.info(f"  [{seg_i+1}/{len(speech_segs)}] {seg_start:.1f}s-{seg_end:.1f}s: {seg_text[:60]}...")
                except Exception as e:
                    logger.warning(f"VAD segment {seg_i} failed: {e}")
            result = {
                "text": " ".join(full_texts).strip(),
                "segments": all_segments,
                "language": "en",
                "n_segments": len(all_segments),
                "vad_segments": [(s,e) for s,e in speech_segs],
                "model_type": wcpp._lib.whisper_model_type_readable(wcpp._ctx).decode() if wcpp._ctx else "unknown",
            }
            # Free intermediate lists
            del full_texts, all_segments, speech_segs
        else:
            # ── RMS-based pre-filter (lightweight VAD, no ML model) ──
            do_hallu = self._get(kwargs,"hallu_filter",True)
            hallu_th = self._get(kwargs,"hallu_threshold",0.6)
            if do_hallu:
                sr = 16000
                window_ms = 30  # 30ms window
                hop_ms = 10     # 10ms step
                win_len = int(sr * window_ms / 1000)
                hop_len = int(sr * hop_ms / 1000)
                threshold = hallu_th * 0.01  # 0.6 -> 0.006
                n_frames = (len(audio_data) - win_len) // hop_len + 1
                
                # Compute RMS per frame
                rms_frames = np.array([
                    np.sqrt(np.mean(audio_data[i*hop_len : i*hop_len + win_len]**2))
                    for i in range(n_frames)
                ])
                
                # Find speech regions (RMS > threshold)
                is_speech = rms_frames > threshold
                if is_speech.any():
                    # Find contiguous regions
                    regions = []
                    in_speech = False
                    start_frame = 0
                    for i in range(n_frames):
                        if is_speech[i] and not in_speech:
                            start_frame = i
                            in_speech = True
                        elif not is_speech[i] and in_speech:
                            # Merge gap < 500ms (50 frames)
                            gap = i - start_frame
                            for j in range(i, min(i + 50, n_frames)):
                                if is_speech[j]:
                                    break
                            else:
                                regions.append((start_frame, i))
                                in_speech = False
                    if in_speech:
                        regions.append((start_frame, n_frames))
                    
                    # Convert to seconds and transcribe per region
                    all_segments, full_texts = [], []
                    for ri, (sf, ef) in enumerate(regions):
                        comfy.model_management.throw_exception_if_processing_interrupted()
                        seg_start = sf * hop_len / sr
                        seg_end = ef * hop_len / sr
                        dur_s = seg_end - seg_start
                        if dur_s < 0.3:  # skip noise spikes < 300ms
                            continue
                        dur_ms = int(dur_s * 1000)
                        if dur_ms < 1000:  # skip <1s
                            continue
                        # Extract audio chunk — NO initial_prompt carry (causes whisper loops)
                        chunk = audio_data[int(seg_start*16000):int(seg_end*16000)]
                        seg_tp = {**tp_base, "offset_ms": 0, "duration_ms": 0, "initial_prompt": None}
                        try:
                            seg_result = wcpp.transcribe(chunk, **seg_tp)
                            seg_text = seg_result.get("text","").strip()
                            if seg_text:
                                full_texts.append(seg_text)
                            for seg in seg_result.get("segments", []):
                                # Shift chunk timestamps (segment + nested words)
                                # to original audio position.
                                seg = shift_segment_timestamps(seg, seg_start)
                                all_segments.append(seg)
                            logger.info(f"  [{ri+1}/{len(regions)}] {seg_start:.1f}s-{seg_end:.1f}s: {seg_text[:60]}...")
                        except Exception as e:
                            logger.warning(f"Region {ri} failed: {e}")
                    result = {
                        "text": " ".join(full_texts).strip(),
                        "segments": all_segments,
                        "language": "en",
                        "n_segments": len(all_segments),
                    }
                else:
                    result = {"text": "", "segments": [], "language": "en", "n_segments": 0}
            else:
                tp = {**tp_base, "offset_ms":0, "duration_ms":0}
                try: result = wcpp.transcribe(audio_data, **tp)
                except Exception as e: logger.error(f"Failed: {e}"); import traceback; traceback.print_exc(); return ("","","","","","","")
        pbar.update(1)
        
        # --- Optional alignment (sherpa-onnx CTC) — default ON ---
        # Skipped if DTW is active (mutually exclusive)
        do_align = self._get(kwargs,"align",True) and not self._get(kwargs,"dtw_token_timestamps",False)
        if do_align:
            if SHERPA_ALIGN_AVAILABLE:
                try:
                    logger.info("Running sherpa-onnx CTC alignment...")
                    from .whispercpp.ext.alignment_sherpa import load_align_model, align
                    am = load_align_model(device)
                    result["segments"] = align(
                        result["segments"], audio_data, am,
                        language=result.get("language","en"),
                    )
                    logger.info("Alignment done")
                except Exception as e:
                    logger.warning(f"Alignment failed: {e}")
            else:
                logger.warning("sherpa-onnx not installed, alignment unavailable")
        pbar.update(1)
        
        # --- Optional diarization (cpp-annote DLL) — default off ---
        if self._get(kwargs,"diarize",False):
            try:
                logger.info("Running speaker diarization via cpp-annote DLL...")
                from .whispercpp.ext.cppannote import diarize as cpp_diarize, CPPANNOTE_AVAILABLE as _cppa
                if not _cppa:
                    raise RuntimeError("cpp-annote not available")
                turns = cpp_diarize(audio_data, sr=16000)
                if turns and (len(turns) > 1 or turns[0].get("speaker",0) != 0):
                    for seg in result.get("segments",[]):
                        seg_start = seg.get("start",0)
                        seg_end = seg.get("end",0)
                        best_spk, best_overlap = 0, 0
                        for t in turns:
                            overlap = max(0, min(seg_end, t.get("end",0)) - max(seg_start, t.get("start",0)))
                            if overlap > best_overlap:
                                best_overlap, best_spk = overlap, t.get("speaker",0)
                        seg["speaker"] = best_spk
                    n_spk = len(set(s.get("speaker",0) for s in result["segments"]))
                    logger.info(f"Diarization done: {n_spk} speaker(s)")
                else:
                    logger.info("Diarization: single speaker")
            except Exception as e:
                logger.warning(f"Diarization skipped: {e}")
        pbar.update(1)
        
        outputs = self._make_outputs(result)
        
        # Cleanup BSRoformer cache to free memory
        try:
            from .whispercpp.ext.uvr import cleanup as uvr_cleanup
            uvr_cleanup()
        except Exception:
            pass
        
        # Cleanup alignment cache
        try:
            from .whispercpp.ext.alignment_sherpa import cleanup as align_cleanup
            align_cleanup()
        except Exception:
            pass
        
        # Cleanup cpp-annote (VAD + diarization) cache
        try:
            from .whispercpp.ext.cppannote import cleanup as cppa_cleanup
            cppa_cleanup()
        except Exception:
            pass
        
        return outputs

    def _make_outputs(self, result):
        full_text = " ".join([s.get("text","").strip() for s in result.get("segments",[])])
        seg_json = json.dumps(result.get("segments",[]), indent=2, ensure_ascii=False)
        segs = result.get("segments",[])
        
        # Auto-detect word-level: use if any segment has words with timestamps
        has_words = any(
            s.get("words") and len(s["words"]) > 0 and "start" in s["words"][0]
            for s in segs
        )
        
        srt = self._segs_to_srt(segs, word_level=has_words)
        vtt = self._segs_to_vtt(segs, word_level=has_words)
        tsv = self._segs_to_tsv(segs, word_level=has_words)
        aud = self._segs_to_aud(segs, word_level=has_words)
        jr = json.dumps(result, indent=2, ensure_ascii=False)
        return (full_text, seg_json, srt, vtt, tsv, aud, jr)

    @staticmethod
    def _ts(sec, fmt="srt"):
        if sec < 0: sec = 0
        h, m = int(sec//3600), int((sec%3600)//60)
        s = sec % 60
        if fmt=="vtt": return f"{h:02d}:{m:02d}:{s:06.3f}"
        ms = int((s-int(s))*1000); return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"

    def _segs_to_srt(self, segs, word_level=False):
        lines = []
        counter = 1
        for seg in segs:
            words = seg.get("words", [])
            if word_level and words:
                for w in words:
                    word_text = w.get("word", "").strip()
                    if not word_text:
                        continue
                    lines.append(str(counter))
                    lines.append(f"{self._ts(w.get('start', 0), 'srt')} --> {self._ts(w.get('end', 0), 'srt')}")
                    lines.append(word_text)
                    lines.append("")
                    counter += 1
            else:
                lines.append(str(counter))
                lines.append(f"{self._ts(seg['start'], 'srt')} --> {self._ts(seg['end'], 'srt')}")
                lines.append(seg.get("text", "").strip())
                lines.append("")
                counter += 1
        return "\n".join(lines)
    def _segs_to_vtt(self, segs, word_level=False):
        lines = ["WEBVTT", ""]
        for seg in segs:
            words = seg.get("words", [])
            if word_level and words:
                for w in words:
                    word_text = w.get("word", "").strip()
                    if not word_text:
                        continue
                    lines.append(f"{self._ts(w.get('start', 0), 'vtt')} --> {self._ts(w.get('end', 0), 'vtt')}")
                    lines.append(word_text)
                    lines.append("")
            else:
                lines.append(f"{self._ts(seg['start'], 'vtt')} --> {self._ts(seg['end'], 'vtt')}")
                lines.append(seg.get("text", "").strip())
                lines.append("")
        return "\n".join(lines)
    def _segs_to_tsv(self, segs, word_level=False):
        lines = ["start\tend\ttext"]
        for seg in segs:
            words = seg.get("words", [])
            if word_level and words:
                for w in words:
                    word_text = w.get("word", "").strip()
                    if not word_text:
                        continue
                    lines.append(f"{w.get('start', 0):.3f}\t{w.get('end', 0):.3f}\t{word_text}")
            else:
                lines.append(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg.get('text', '').strip()}")
        return "\n".join(lines)
    def _segs_to_aud(self, segs, word_level=False):
        lines = []
        for seg in segs:
            speaker = seg.get("speaker", 0)
            words = seg.get("words", [])
            if word_level and words:
                for w in words:
                    word_text = w.get("word", "").strip()
                    if not word_text:
                        continue
                    lines.append(f"{w.get('start', 0):.3f},{w.get('end', 0):.3f},{word_text},{speaker}")
            else:
                lines.append(f"{seg['start']:.3f},{seg['end']:.3f},{seg.get('text', '').strip()},{speaker}")
        return "\n".join(lines)

NODE_CLASS_MAPPINGS = {"WhisperCPPNode": WhisperCPPNode}
NODE_DISPLAY_NAME_MAPPINGS = {"WhisperCPPNode": "WhisperCPP Transcription"}
