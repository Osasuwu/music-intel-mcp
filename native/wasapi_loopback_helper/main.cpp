// wasapi_loopback_helper.exe — minimal native WASAPI per-process loopback
// capture (issue #124). Activates VAD\Process_Loopback via
// ActivateAudioInterfaceAsync scoped to a target PID, then streams raw
// 16-bit PCM to stdout until max_seconds elapses or the process is killed.
//
// Exists because comtypes (Python) never receives the ActivateCompleted
// callback for this specific API — the callback is delivered by audiosrv
// via genuine out-of-process RPC, and a hand-rolled native repro proved
// that path works correctly outside comtypes. See PR description / issue
// #124 for the full debugging trail. This file is the proven-working
// min_repro.cpp extended with real IAudioCaptureClient capture, kept
// deliberately free of WIL/WRL/Media Foundation — a synchronous blocking
// loop on a single Win32 event is all a console helper needs.
//
// Wire protocol (stdout, binary):
//   1. One 16-byte header: struct { uint32 status; uint32 sampleRate;
//      uint32 channels; uint32 bitsPerSample; }. status == 0 means success
//      and sampleRate/channels/bitsPerSample describe every PCM byte that
//      follows. status != 0 is an HRESULT (or 0xFFFFFFFF for "activation
//      timed out") and no PCM follows — the process then exits non-zero.
//   2. On success: a continuous raw interleaved PCM byte stream (no further
//      framing) until max_seconds elapses or the process is terminated.
//
// Audio is never written to any file — stdout is the only sink (AC2).

#include <windows.h>
#include <audioclientactivationparams.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <objbase.h>
#include <io.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <vector>

namespace {

class Handler : public IActivateAudioInterfaceCompletionHandler {
public:
    Handler() { hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr); }
    ~Handler() {
        if (hEvent) CloseHandle(hEvent);
        if (audioClient) audioClient->Release();
    }

    // IUnknown
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override {
        if (riid == __uuidof(IUnknown) ||
            riid == __uuidof(IActivateAudioInterfaceCompletionHandler) ||
            riid == __uuidof(IAgileObject)) {
            *ppv = static_cast<IActivateAudioInterfaceCompletionHandler*>(this);
            AddRef();
            return S_OK;
        }
        *ppv = nullptr;
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef() override { return ++refCount; }
    ULONG STDMETHODCALLTYPE Release() override {
        ULONG r = --refCount;
        if (r == 0) delete this;
        return r;
    }

    // IActivateAudioInterfaceCompletionHandler — delivered on an MTA worker
    // thread by audiosrv via out-of-process RPC. No message pump needed:
    // the caller blocks on hEvent with WaitForSingleObject.
    HRESULT STDMETHODCALLTYPE ActivateCompleted(IActivateAudioInterfaceAsyncOperation* operation) override {
        HRESULT hrActivate = E_UNEXPECTED;
        IUnknown* punk = nullptr;
        operation->GetActivateResult(&hrActivate, &punk);
        activateResult = hrActivate;
        if (SUCCEEDED(hrActivate) && punk) {
            punk->QueryInterface(__uuidof(IAudioClient), reinterpret_cast<void**>(&audioClient));
        }
        if (punk) punk->Release();
        SetEvent(hEvent);
        return S_OK;
    }

    HANDLE hEvent = nullptr;
    HRESULT activateResult = E_UNEXPECTED;
    IAudioClient* audioClient = nullptr;
    std::atomic<ULONG> refCount{1};
};

void writeHeaderAndFlush(uint32_t status, uint32_t sampleRate, uint32_t channels, uint32_t bitsPerSample) {
    uint32_t header[4] = {status, sampleRate, channels, bitsPerSample};
    fwrite(header, sizeof(header), 1, stdout);
    fflush(stdout);
}

int fail(uint32_t hr) {
    writeHeaderAndFlush(hr, 0, 0, 0);
    return 1;
}

}  // namespace

int wmain(int argc, wchar_t* argv[]) {
    if (argc < 3) {
        fwprintf(stderr, L"Usage: wasapi_loopback_helper.exe <pid> <max_seconds> [includetree|excludetree]\n");
        return 2;
    }
    DWORD targetPid = wcstoul(argv[1], nullptr, 0);
    double maxSeconds = _wtof(argv[2]);
    bool includeTree = true;
    if (argc >= 4 && wcscmp(argv[3], L"excludetree") == 0) {
        includeTree = false;
    }

    // stdout carries binary PCM — force binary mode so the CRT never
    // rewrites 0x0A bytes in the audio stream to 0x0D 0x0A.
    _setmode(_fileno(stdout), _O_BINARY);

    HRESULT hrInit = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hrInit) && hrInit != RPC_E_CHANGED_MODE) {
        return fail(static_cast<uint32_t>(hrInit));
    }

    AUDIOCLIENT_ACTIVATION_PARAMS activationParams = {};
    activationParams.ActivationType = AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK;
    activationParams.ProcessLoopbackParams.TargetProcessId = targetPid;
    activationParams.ProcessLoopbackParams.ProcessLoopbackMode = includeTree
        ? PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE
        : PROCESS_LOOPBACK_MODE_EXCLUDE_TARGET_PROCESS_TREE;

    PROPVARIANT activateParams = {};
    activateParams.vt = VT_BLOB;
    activateParams.blob.cbSize = sizeof(activationParams);
    activateParams.blob.pBlobData = reinterpret_cast<BYTE*>(&activationParams);

    Handler* handler = new Handler();
    IActivateAudioInterfaceAsyncOperation* asyncOp = nullptr;

    HRESULT hr = ActivateAudioInterfaceAsync(
        VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK,
        __uuidof(IAudioClient),
        &activateParams,
        handler,
        &asyncOp);
    if (FAILED(hr)) {
        int rc = fail(static_cast<uint32_t>(hr));
        handler->Release();
        return rc;
    }

    DWORD waitResult = WaitForSingleObject(handler->hEvent, 30000);
    if (waitResult != WAIT_OBJECT_0) {
        int rc = fail(0xFFFFFFFFu);  // sentinel: activation timed out
        if (asyncOp) asyncOp->Release();
        handler->Release();
        return rc;
    }
    if (FAILED(handler->activateResult) || !handler->audioClient) {
        int rc = fail(static_cast<uint32_t>(handler->activateResult));
        if (asyncOp) asyncOp->Release();
        handler->Release();
        return rc;
    }

    IAudioClient* audioClient = handler->audioClient;
    audioClient->AddRef();  // keep our own ref independent of handler's dtor

    WAVEFORMATEX format = {};
    format.wFormatTag = WAVE_FORMAT_PCM;
    format.nChannels = 2;
    format.nSamplesPerSec = 44100;
    format.wBitsPerSample = 16;
    format.nBlockAlign = static_cast<WORD>(format.nChannels * format.wBitsPerSample / 8);
    format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign;

    hr = audioClient->Initialize(
        AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK | AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM,
        0, 0, &format, nullptr);
    if (FAILED(hr)) {
        int rc = fail(static_cast<uint32_t>(hr));
        audioClient->Release();
        if (asyncOp) asyncOp->Release();
        handler->Release();
        return rc;
    }

    HANDLE sampleEvent = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    hr = audioClient->SetEventHandle(sampleEvent);
    if (FAILED(hr)) {
        int rc = fail(static_cast<uint32_t>(hr));
        CloseHandle(sampleEvent);
        audioClient->Release();
        if (asyncOp) asyncOp->Release();
        handler->Release();
        return rc;
    }

    IAudioCaptureClient* captureClient = nullptr;
    hr = audioClient->GetService(__uuidof(IAudioCaptureClient), reinterpret_cast<void**>(&captureClient));
    if (FAILED(hr)) {
        int rc = fail(static_cast<uint32_t>(hr));
        CloseHandle(sampleEvent);
        audioClient->Release();
        if (asyncOp) asyncOp->Release();
        handler->Release();
        return rc;
    }

    hr = audioClient->Start();
    if (FAILED(hr)) {
        int rc = fail(static_cast<uint32_t>(hr));
        captureClient->Release();
        CloseHandle(sampleEvent);
        audioClient->Release();
        if (asyncOp) asyncOp->Release();
        handler->Release();
        return rc;
    }

    // Activation + Initialize + Start all succeeded — announce the format
    // and start streaming. Everything before this point is silent on stdout.
    writeHeaderAndFlush(0, format.nSamplesPerSec, format.nChannels, format.wBitsPerSample);

    std::vector<BYTE> zeros;
    ULONGLONG startTick = GetTickCount64();
    ULONGLONG maxMs = static_cast<ULONGLONG>(maxSeconds * 1000.0);

    while (GetTickCount64() - startTick < maxMs) {
        // 200ms poll period so the max-duration deadline is still honored
        // even if the stream goes quiet (e.g. target process pauses).
        DWORD w = WaitForSingleObject(sampleEvent, 200);
        if (w != WAIT_OBJECT_0) {
            continue;
        }

        UINT32 framesAvailable = 0;
        while (SUCCEEDED(captureClient->GetNextPacketSize(&framesAvailable)) && framesAvailable > 0) {
            BYTE* data = nullptr;
            DWORD flags = 0;
            hr = captureClient->GetBuffer(&data, &framesAvailable, &flags, nullptr, nullptr);
            if (FAILED(hr)) {
                break;
            }

            DWORD bytes = framesAvailable * format.nBlockAlign;
            if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
                // Silent packets aren't guaranteed a valid data pointer —
                // the client is expected to synthesize silence itself.
                if (zeros.size() < bytes) {
                    zeros.assign(bytes, 0);
                }
                fwrite(zeros.data(), 1, bytes, stdout);
            } else {
                fwrite(data, 1, bytes, stdout);
            }
            captureClient->ReleaseBuffer(framesAvailable);
        }
        fflush(stdout);
    }

    audioClient->Stop();
    captureClient->Release();
    CloseHandle(sampleEvent);
    audioClient->Release();
    if (asyncOp) asyncOp->Release();
    handler->Release();
    CoUninitialize();
    return 0;
}
