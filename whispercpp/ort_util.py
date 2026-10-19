"""Shared ONNX Runtime DLL helpers (issue #17).

The process can only use ONE ``onnxruntime.dll``: the Windows loader resolves
that name to whichever copy is already loaded, so the FIRST copy that gets in
decides the ONNX Runtime API available to every later dependent (whisper
cpp-annote, sherpa-onnx, ...). These helpers read DLL versions and list the
copies already loaded so callers can preload the newest one and detect a
poisoned process before native code crashes it.
"""

import ctypes
import re
import sys
from pathlib import Path


def win_file_version(path):
    """Read a DLL's FileVersion via the Win32 version API, or None."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    ver = ctypes.WinDLL("version.dll")
    ver.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    ver.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    ver.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    ver.GetFileVersionInfoW.restype = wintypes.BOOL
    ver.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)]
    ver.VerQueryValueW.restype = wintypes.DWORD

    dummy = wintypes.DWORD()
    size = ver.GetFileVersionInfoSizeW(str(path), ctypes.byref(dummy))
    if not size:
        return None
    buf = ctypes.create_string_buffer(size)
    if not ver.GetFileVersionInfoW(str(path), 0, size, buf):
        return None
    root = ctypes.c_void_p()
    length = wintypes.UINT()
    if not ver.VerQueryValueW(buf, "\\", ctypes.byref(root), ctypes.byref(length)):
        return None

    class VS_FIXEDFILEINFO(ctypes.Structure):
        _fields_ = [
            ("dwSignature", wintypes.DWORD), ("dwStrucVersion", wintypes.DWORD),
            ("dwFileVersionMS", wintypes.DWORD), ("dwFileVersionLS", wintypes.DWORD),
            ("dwProductVersionMS", wintypes.DWORD), ("dwProductVersionLS", wintypes.DWORD),
            ("dwFileFlagsMask", wintypes.DWORD), ("dwFileFlags", wintypes.DWORD),
            ("dwFileOS", wintypes.DWORD), ("dwFileType", wintypes.DWORD),
            ("dwFileSubtype", wintypes.DWORD), ("dwFileDateMS", wintypes.DWORD),
            ("dwFileDateLS", wintypes.DWORD),
        ]

    ffi = ctypes.cast(root, ctypes.POINTER(VS_FIXEDFILEINFO)).contents
    ms, ls = ffi.dwFileVersionMS, ffi.dwFileVersionLS
    return (ms >> 16, ms & 0xFFFF, ls >> 16, ls & 0xFFFF)


def parse_so_version(name):
    """Parse 'libonnxruntime.so.1.28.2' -> (1, 28, 2, 0), else None."""
    m = re.search(r"libonnxruntime\.so\.(\d+)\.(\d+)\.(\d+)$", name)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), 0) if m else None


def dll_file_version(path):
    """Version tuple of a bundled onnxruntime library, or None if unknown."""
    path = Path(path)
    if sys.platform == "win32":
        return win_file_version(path) if path.exists() else None
    if path.exists():
        return parse_so_version(path.name)
    return None


def sherpa_bundle_dll():
    """Path of the onnxruntime lib bundled with sherpa-onnx, or None.

    Uses find_spec so sherpa_onnx itself is not imported here.
    """
    import importlib.util
    try:
        spec = importlib.util.find_spec("sherpa_onnx")
        if not (spec and spec.submodule_search_locations):
            return None
        for loc in spec.submodule_search_locations:
            lib = Path(loc) / "lib"
            if sys.platform == "win32":
                p = lib / "onnxruntime.dll"
                if p.exists():
                    return p
            elif lib.is_dir():
                for f in sorted(lib.glob("libonnxruntime.so.*")):
                    return f
    except Exception:
        return None
    return None


def pick_newest_ort(candidates, version_fn=dll_file_version):
    """Existing candidate with the highest version (None if none exist)."""
    best_path, best_key = None, None
    for p in candidates:
        if not p or not Path(p).exists():
            continue
        key = version_fn(p) or (0,)
        if best_key is None or key > best_key:
            best_path, best_key = Path(p), key
    return best_path


def loaded_ort_modules():
    """[(path, version)] of onnxruntime.dll copies loaded in this process.

    Windows only; returns [] elsewhere or when enumeration fails.
    """
    if sys.platform != "win32":
        return []
    try:
        from ctypes import wintypes

        psapi = ctypes.WinDLL("psapi")
        k32 = ctypes.WinDLL("kernel32")
        psapi.EnumProcessModules.restype = wintypes.BOOL
        psapi.EnumProcessModules.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p), wintypes.DWORD,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        psapi.GetModuleFileNameExW.restype = wintypes.DWORD
        psapi.GetModuleFileNameExW.argtypes = [
            wintypes.HANDLE, ctypes.c_void_p, wintypes.LPWSTR, wintypes.DWORD,
        ]
        handle = k32.GetCurrentProcess()
        needed = ctypes.c_ulong()
        head = ctypes.c_void_p()
        if not psapi.EnumProcessModules(handle, ctypes.byref(head), ctypes.sizeof(head), ctypes.byref(needed)):
            return []
        count = needed.value // ctypes.sizeof(ctypes.c_void_p)
        arr = (ctypes.c_void_p * count)()
        if not psapi.EnumProcessModules(handle, arr, ctypes.sizeof(arr), ctypes.byref(needed)):
            return []
        out = []
        buf = ctypes.create_unicode_buffer(1024)
        for i in range(count):
            psapi.GetModuleFileNameExW(handle, arr[i], buf, 1024)
            path = buf.value or ""
            if path and Path(path).name.lower() == "onnxruntime.dll":
                out.append((path, win_file_version(path)))
        return out
    except Exception:
        return []
