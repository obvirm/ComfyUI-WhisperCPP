# ComfyUI-WhisperCPP

ComfyUI custom node for speech-to-text using **whisper.cpp** (C API via ctypes).  
Full pipeline: ASR → Vocal Separation → VAD → Alignment → Diarization — **zero whisperx**.

## Screenshots

![Node](picture/new.png)
![Advance CPP](picture/new1.png)
![Advance EXT](picture/new2.png)

## Features

- **whisper.cpp native** — DLL + ctypes binding directly to C API, no Python wrapper
- **All `whisper_full_params`** — every parameter exposed as a node widget
- **Two advance toggles** — `show_advance_cpp` (core whisper params) + `show_advance_ext` (UVR/alignment/diarization detail)
- **7 output sockets** — text, segments_json, srt, vtt, tsv, aud, json_result (no file I/O)
- **Auto-download** — DLLs downloaded from GitHub Releases on first use (no manual build needed)
- **GPU acceleration** — Vulkan, OpenCL, CUDA, Metal (auto-detected at runtime)
- **UVR vocal separation** — BSRoformer.cpp DLL (GGUF model, no temp files)
- **VAD** — cpp-annote (community-1 segmentation model, DLL + onnxruntime)
- **Pre-filter** — RMS-based energy filter to prevent hallucinations on silence
- **Alignment** — sherpa-onnx CTC (zipformer, word-level timestamps)
- **Diarization** — cpp-annote DLL (no Python dependencies)
- **Cross-platform** — Windows/MSVC, Linux/GCC, macOS/Clang (DLL/so/dylib)

## Quick Start

```bash
cd ComfyUI/custom_nodes/
git clone --recurse-submodules https://github.com/obvirm/ComfyUI-WhisperCPP
cd ComfyUI-WhisperCPP
pip install -r requirements.txt
```

That's it! DLLs are **automatically downloaded** from GitHub Releases on first use. No build step required.

### Build from Source (Optional)

If you want to build the DLLs yourself (e.g., for GPU-specific optimizations):

```bash
# whisper.cpp — auto-detect GPU backends
python build_whisper_cpp.py

# BSRoformer DLL
python build_bs_roformer.py

# cpp-annote DLL
python build_cpp_annote.py
```

#### Build Options

```bash
# whisper.cpp
python build_whisper_cpp.py --gpu vulkan   # Force Vulkan
python build_whisper_cpp.py --gpu cuda     # Force CUDA
python build_whisper_cpp.py --gpu cpu      # CPU only

# BSRoformer
python build_bs_roformer.py --gpu          # Auto-detect GPU
python build_bs_roformer.py                # CPU only

# cpp-annote
python build_cpp_annote.py --cuda          # CUDA ONNX Runtime
python build_cpp_annote.py --gpu           # Auto-detect GPU
python build_cpp_annote.py                 # CPU only
```

## Modules

| Module | Purpose | Binding | File |
|---|---|---|---|
| **whisper.cpp** | ASR (speech-to-text) | `whisper_lib.py` | `whisper.dll` + ggml DLLs |
| **BSRoformer.cpp** | UVR vocal separation | `bs_roformer_lib.py` | `bs_roformer.dll` |
| **cpp-annote** | VAD + Diarization | `cpp_annote_lib.py` | `cpp_annote.dll` + onnxruntime.dll |

All modules follow the same pattern: **C API header → shared library → ctypes binding**.

## Pipeline

```
Audio Input
  │
  ├─ [UVR] BSRoformer.dll → vocal separation (optional)
  │    16kHz → Resample(44100) → DLL → mono → Resample(16000)
  │
  ├─ [Pre-filter] RMS energy → speech detection (optional)
  │    Only transcribe sections with audio energy > threshold
  │
  ├─ [VAD] cpp-annote.dll → speech segmentation (optional)
  │
  ├─ whisper.dll → transcribe (per-segment if VAD/pre-filter on)
  │
  ├─ [Alignment] sherpa-onnx CTC → word timestamps (default ON)
  │
  └─ [Diarization] cpp-annote DLL → speaker labels (optional)
       │
       └─ 7 output sockets
```

## Auto-Download

On first use, each module automatically downloads its DLLs from GitHub Releases:

```
1. _find_library() → DLL not found
2. gpu_detect.detect_gpu() → NVIDIA/Vulkan/OpenCL/CPU
3. auto_download.download_module("whisper", dir, has_gpu)
   → Download whisper.dll, ggml-base.dll, ggml-cpu.dll, ggml.dll
   → (GPU) also download ggml-vulkan.dll
4. _find_library() → found!
```

- **Version is dynamic** — reads from `pyproject.toml` (no manual updates)
- **GPU-aware** — downloads Vulkan/OpenCL DLLs when GPU is detected
- **Skip existing** — won't re-download if DLLs are already present
- **All platforms** — Windows (`.dll`), Linux (`.so`), macOS (`.dylib`)

## Usage

1. Add **WhisperCPPNode**
2. Connect audio source to `audio` socket
3. Select model (auto-downloads on first use)
4. Set `language` — code (`ja`), English name (`Japanese`), `auto`, or `None` for auto-detect. Foreign labels (e.g. `日本語 (Japanese)` from other extensions) are normalized automatically (issue #16)
5. Enable `hallu_filter` (default ON) for RMS-based pre-filter
6. Set `vad=true` for active speech detection
7. Set `separate_vocals=true` for vocal separation
8. Enable `show_advance_cpp` / `show_advance_ext` for full parameter access

### Output Sockets

| Socket | Format | Content |
|---|---|---|
| `text` | STRING | Full transcript |
| `segments_json` | STRING (JSON) | Segments with timestamps |
| `srt` | STRING | SubRip subtitle |
| `vtt` | STRING | WebVTT subtitle |
| `tsv` | STRING | Tab-separated values |
| `aud` | STRING | Speaker-labeled transcript |
| `json_result` | STRING (JSON) | Full result metadata |

## Models

| Model | Location | Size | Auto-Download |
|---|---|---|---|
| Whisper GGML | `ComfyUI/models/whispercpp/` | 1.6 GB | ✅ |
| UVR GGUF | `ComfyUI/models/uvr/` | 251 MB | ✅ |
| Alignment CTC | `ComfyUI/models/alignment/sherpa/` | 383 MB | ✅ |
| cpp-annote ONNX | `cpp-annote/artifacts/` | 32 MB | In repo |

## GPU Backends

| Backend | Windows | Linux | macOS |
|---------|---------|-------|-------|
| Vulkan  | ✅ | ✅ | ✅ |
| OpenCL  | ✅ | ✅ | - |
| CUDA    | ✅* | ✅ | - |
| Metal   | - | - | ✅ |

*CUDA requires CUDA Toolkit installed.

## Architecture

```
ComfyUI-WhisperCPP/
├── __init__.py                 # Register node
├── whispercpp_node.py          # Class WhisperCPPNode
├── js/whispercpp_node.js       # Frontend toggle (advance sections)
│
├── whispercpp/
│   ├── whisper_lib.py          # ctypes → whisper.dll
│   ├── gpu_detect.py           # GPU detection (NVIDIA/Vulkan/OpenCL)
│   ├── auto_download.py        # Download DLLs from GitHub Releases
│   ├── model.py                # Model download manager
│   ├── audio.py                # Audio processor
│   └── ext/
│       ├── uvr.py              # Vocal separation (BSRoformer DLL)
│       ├── bs_roformer_lib.py  # ctypes → bs_roformer.dll
│       ├── alignment_sherpa.py # CTC alignment
│       ├── cpp_annote_lib.py   # ctypes → cpp_annote.dll
│       └── cppannote.py        # VAD + diarization wrapper
│
├── whisper.cpp/                # git submodule
├── bs_roformer.cpp/            # git submodule
├── cpp-annote/                 # git submodule
│
├── build_whisper_cpp.py        # Cross-platform build
├── build_bs_roformer.py        # Cross-platform build
├── build_cpp_annote.py         # Cross-platform build
│
├── AGENTS.md                   # AI agent rules
├── ROADMAP.md                  # Future plans
└── MEMD.md                     # Project memory
```

## License

Apache 2.0
