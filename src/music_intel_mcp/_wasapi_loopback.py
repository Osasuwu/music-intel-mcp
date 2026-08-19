"""Raw ctypes/comtypes interop for Windows per-process WASAPI loopback capture.

Isolated in its own module (imported lazily by :mod:`capture`, never at package
import time) because it is Windows-only, hardware-facing, and has no meaningful
in-process fake: :class:`~music_intel_mcp.capture.WasapiProcessLoopbackCapture`
is validated against a real playback session (issue #124 AC1), not covered by
this repo's unit test suite. Requires Windows 10 2004+ (build 19041+) for the
``AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK`` activation path.

References: the Process Loopback API shipped in the ``mmdeviceapi.h`` /
``audioclientactivationparams.h`` Windows SDK headers (no Python wheel wraps
this specifically — pyaudiowpatch/soundcard only expose device-level loopback).
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

import comtypes
import numpy as np
from comtypes import GUID, COMObject, IUnknown

# -- constants ------------------------------------------------------------- #

AUDIOCLIENT_ACTIVATION_TYPE_DEFAULT = 0
AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1

PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0

VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK = "VAD\\Process_Loopback"

AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_SHAREMODE_SHARED = 0

VT_BLOB = 65

IID_IAudioClient = GUID("{1CB9AD4C-DBFA-4c32-B178-C2F568A703B2}")


class WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ("wFormatTag", wintypes.WORD),
        ("nChannels", wintypes.WORD),
        ("nSamplesPerSec", wintypes.DWORD),
        ("nAvgBytesPerSec", wintypes.DWORD),
        ("nBlockAlign", wintypes.WORD),
        ("wBitsPerSample", wintypes.WORD),
        ("cbSize", wintypes.WORD),
    ]


WAVE_FORMAT_IEEE_FLOAT = 3


class AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS(ctypes.Structure):
    _fields_ = [
        ("TargetProcessId", wintypes.DWORD),
        ("ProcessLoopbackMode", ctypes.c_int),
    ]


class _ACTIVATION_PARAMS_UNION(ctypes.Union):
    _fields_ = [("ProcessLoopbackParams", AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS)]


class AUDIOCLIENT_ACTIVATION_PARAMS(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("ActivationType", ctypes.c_int), ("u", _ACTIVATION_PARAMS_UNION)]


class _BLOB(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.ULONG), ("pBlobData", ctypes.POINTER(ctypes.c_byte))]


class PROPVARIANT(ctypes.Structure):
    """Minimal VT_BLOB-only PROPVARIANT — enough to carry the activation params."""

    _fields_ = [
        ("vt", wintypes.USHORT),
        ("wReserved1", wintypes.USHORT),
        ("wReserved2", wintypes.USHORT),
        ("wReserved3", wintypes.USHORT),
        ("blob", _BLOB),
        ("_pad", ctypes.c_byte * 8),
    ]


class IActivateAudioInterfaceAsyncOperation(IUnknown):
    _iid_ = GUID("{72A22D78-CDE4-431D-B8CC-843A71199B6D}")
    _methods_ = []  # only GetActivateResult is used, called via raw vtable below


class IActivateAudioInterfaceCompletionHandler(COMObject):
    _com_interfaces_ = [IUnknown]
    _iid_ = GUID("{41D949AB-9862-444A-80F6-C261334DA5EB}")

    def __init__(self) -> None:
        super().__init__()
        self.done = False
        self.hresult = None
        self.audio_client = None

    def IActivateAudioInterfaceCompletionHandler_ActivateCompleted(self, operation):
        hr_activate = wintypes.HRESULT()
        punk = ctypes.POINTER(IUnknown)()
        # GetActivateResult(HRESULT*, IUnknown**) is vtable slot 3 (after
        # QueryInterface/AddRef/Release) on IActivateAudioInterfaceAsyncOperation.
        vtbl = ctypes.cast(operation, ctypes.POINTER(ctypes.c_void_p * 4))
        get_activate_result = ctypes.WINFUNCTYPE(
            ctypes.HRESULT,
            ctypes.c_void_p,
            ctypes.POINTER(wintypes.HRESULT),
            ctypes.POINTER(ctypes.POINTER(IUnknown)),
        )(vtbl.contents[3])
        get_activate_result(operation, ctypes.byref(hr_activate), ctypes.byref(punk))
        self.hresult = hr_activate.value
        if punk:
            self.audio_client = punk.QueryInterface(comtypes.gen.IUnknown)
        self.done = True
        return 0


_mmdevapi = ctypes.WinDLL("Mmdevapi.dll")
_ActivateAudioInterfaceAsync = _mmdevapi.ActivateAudioInterfaceAsync
_ActivateAudioInterfaceAsync.restype = ctypes.HRESULT
_ActivateAudioInterfaceAsync.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(GUID),
    ctypes.POINTER(PROPVARIANT),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]


def _build_activation_params(target_pid: int) -> tuple[PROPVARIANT, AUDIOCLIENT_ACTIVATION_PARAMS]:
    params = AUDIOCLIENT_ACTIVATION_PARAMS()
    params.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK
    params.ProcessLoopbackParams.TargetProcessId = target_pid
    params.ProcessLoopbackParams.ProcessLoopbackMode = (
        PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
    )

    pv = PROPVARIANT()
    pv.vt = VT_BLOB
    pv.blob.cbSize = ctypes.sizeof(params)
    pv.blob.pBlobData = ctypes.cast(ctypes.byref(params), ctypes.POINTER(ctypes.c_byte))
    return pv, params  # keep `params` alive alongside the PROPVARIANT that points into it


def activate_process_loopback(*, target_pid: int, sample_rate: int, channels: int):
    """Activate a per-process loopback IAudioClient scoped to ``target_pid`` and
    return ``(audio_client, capture_client)`` COM pointers, initialized in
    shared-mode loopback and started. Raises ``OSError`` on any HRESULT failure."""
    pv, _keepalive = _build_activation_params(target_pid)
    handler = IActivateAudioInterfaceCompletionHandler()
    op = ctypes.c_void_p()

    hr = _ActivateAudioInterfaceAsync(
        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        ctypes.byref(IID_IAudioClient),
        ctypes.byref(pv),
        ctypes.cast(handler.this, ctypes.c_void_p),
        ctypes.byref(op),
    )
    if hr != 0:
        raise OSError(f"ActivateAudioInterfaceAsync failed: hresult=0x{hr & 0xFFFFFFFF:08X}")

    deadline = time.monotonic() + 5.0
    while not handler.done and time.monotonic() < deadline:
        time.sleep(0.01)
    if not handler.done:
        raise TimeoutError("ActivateAudioInterfaceAsync did not complete in time")
    if handler.hresult:
        raise OSError(f"loopback activation failed: hresult=0x{handler.hresult & 0xFFFFFFFF:08X}")

    audio_client = handler.audio_client
    if audio_client is None:
        raise OSError("loopback activation returned no IAudioClient")

    wfx = WAVEFORMATEX()
    wfx.wFormatTag = WAVE_FORMAT_IEEE_FLOAT
    wfx.nChannels = channels
    wfx.nSamplesPerSec = sample_rate
    wfx.wBitsPerSample = 32
    wfx.nBlockAlign = channels * wfx.wBitsPerSample // 8
    wfx.nAvgBytesPerSec = sample_rate * wfx.nBlockAlign
    wfx.cbSize = 0

    # IAudioClient::Initialize(shareMode, streamFlags, hnsBufferDuration,
    #   hnsPeriodicity, pFormat, audioSessionGuid)
    initialize = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_longlong,
        ctypes.c_longlong,
        ctypes.POINTER(WAVEFORMATEX),
        ctypes.POINTER(GUID),
    )(_vtbl_slot(audio_client, 3))
    hr = initialize(
        audio_client.this if hasattr(audio_client, "this") else audio_client,
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK,
        10_000_000,  # 1s buffer, in 100ns units
        0,
        ctypes.byref(wfx),
        None,
    )
    if hr != 0:
        raise OSError(f"IAudioClient::Initialize failed: hresult=0x{hr & 0xFFFFFFFF:08X}")

    get_service = ctypes.WINFUNCTYPE(
        ctypes.HRESULT, ctypes.c_void_p, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)
    )(_vtbl_slot(audio_client, 14))
    iid_capture_client = GUID("{C8ADBD64-E71E-48a0-A4DE-185C395CD317}")
    capture_client = ctypes.c_void_p()
    hr = get_service(
        audio_client.this if hasattr(audio_client, "this") else audio_client,
        ctypes.byref(iid_capture_client),
        ctypes.byref(capture_client),
    )
    if hr != 0:
        raise OSError(f"GetService(IAudioCaptureClient) failed: hresult=0x{hr & 0xFFFFFFFF:08X}")

    start = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(_vtbl_slot(audio_client, 10))
    hr = start(audio_client.this if hasattr(audio_client, "this") else audio_client)
    if hr != 0:
        raise OSError(f"IAudioClient::Start failed: hresult=0x{hr & 0xFFFFFFFF:08X}")

    return audio_client, capture_client


def _vtbl_slot(punk, index: int):
    vtbl = ctypes.cast(
        punk.this if hasattr(punk, "this") else punk, ctypes.POINTER(ctypes.c_void_p * 32)
    )
    return vtbl.contents[index]


def read_available(
    capture_client, *, duration_s: float, sample_rate: int, channels: int
) -> np.ndarray:
    """Poll ``IAudioCaptureClient::GetBuffer`` for up to ``duration_s`` seconds,
    returning whatever float32 PCM arrived (may be shorter than requested if
    the source produced less audio in that window)."""
    get_buffer = ctypes.WINFUNCTYPE(
        ctypes.HRESULT,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(wintypes.UINT),
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
        ctypes.c_void_p,
    )(_vtbl_slot(capture_client, 3))
    release_buffer = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, wintypes.UINT)(
        _vtbl_slot(capture_client, 4)
    )

    chunks: list[np.ndarray] = []
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        data = ctypes.POINTER(ctypes.c_float)()
        n_frames = wintypes.UINT()
        flags = wintypes.DWORD()
        hr = get_buffer(
            capture_client,
            ctypes.byref(data),
            ctypes.byref(n_frames),
            ctypes.byref(flags),
            None,
            None,
        )
        if hr != 0:
            time.sleep(0.005)
            continue
        if n_frames.value:
            arr = np.ctypeslib.as_array(data, shape=(n_frames.value * channels,)).copy()
            chunks.append(arr.reshape(-1, channels))
            release_buffer(capture_client, n_frames.value)
        else:
            time.sleep(0.005)

    if not chunks:
        return np.zeros((0, channels), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def stop(audio_client) -> None:
    stop_fn = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p)(_vtbl_slot(audio_client, 11))
    stop_fn(audio_client.this if hasattr(audio_client, "this") else audio_client)
