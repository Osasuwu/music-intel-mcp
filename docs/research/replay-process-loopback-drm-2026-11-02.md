---
topic: replay-process-loopback-drm
tags: area:engine, research
source_provenance: /research invocation from grill of Osasuwu/music-intel-mcp#262
status: open
audit: 2026-11-02 — 123 claims: 74 SUPPORTED, 38 PARTIAL, 1 UNSUPPORTED, 10 UNCHECKABLE
---

## Question

Узкий проход для research-гейта grill по Osasuwu/music-intel-mcp#262. Вопросы: (1) работает ли Windows per-process WASAPI loopback (ActivateAudioInterfaceAsync, PROCESS_LOOPBACK) на браузере Chromium/Edge/Firefox, играющем Widevine-защищённый Spotify web player, и не глушится ли аудио; (2) почему process loopback может давать уровень на 16 дБ ниже на одном эндпоинте (JBL Quantum 800 Game), чем на другом, при том что mute и громкость эндпоинта на захват не влияют: APO/SFX, spatial audio, sample-rate, микс; (3) работают ли Windows per-app output (AudioPolicyConfig) и setSinkId для такой вкладки. Артефакт: docs/research/replay-process-loopback-drm-*.md. Закончить проход без кода, вернуть выводы с источниками.

Sub-questions: Q1 process loopback against Widevine audio in a browser; Q2 causes of the 16 dB gap between endpoints; Q3 per-app output and setSinkId for such a tab.

No earlier report exists: the repo has no `docs/research/` directory on `main`.

## Summary

**Recommendation.** Keep the process-loopback route for #262, and make its first step one bench capture on a real Spotify web-player tab, because no source reports a direct test of process loopback on a Widevine stream [C132]. Route replay through a separate browser binary on the Windows per-app setting, verify that route at every start, and do not build on setSinkId. Explain the 16 dB gap before the #169 gate runs on real captures.

**Q1.** No blocking mechanism was found and real audio is plausible, but the case is architectural and unconfirmed. On desktop Widevine leaves audio decoding to the browser [C101], and in the paper's experiments audio goes through a decrypt-only step at the lowest security level [C102]. A Mozilla engineer, reviewing a MediaElementSource test, states that the EME capture policy blocks video and not audio [C107]. Chromium's refusal of captureStream on an EME element is an in-page rule above the OS mixer [C109], [C111]. Two Chromium render sources hold no protection reference [C116], and one Spotify configuration asks for software-level audio robustness, though the configuration chosen in a live session was not observed [C339].

The counter-claim stands unresolved: Microsoft says a trusted audio driver keeps loopback from capturing protected content [C112]. That text speaks of DRM-protected content and trusted drivers, names no browser key system and does not name process loopback [C112], [C113], [C115], so it is neither shown to reach Widevine browser audio nor shown to stop short of it [C117]. Edge or any Media Foundation path stays open [C132], [C133]. A silent test does not by itself mean DRM: a target with no render streams yields silence [C119], one tool's missing flag silenced all streams [C120], and another tool left process loopback after silent failures on sandboxed apps, with no cause given [C121]. The capture root must be the top browser process, which is prior knowledge here [C134].

**Q2.** The cause is not isolated. The gap is 15.99 dB, a factor of 0.1586, computed from a stated 0.0353 with no retained log [C221], and on those values it decides whether the silence gate passes the capture [C245]. Mute and volume having no effect fits a tap upstream of endpoint volume and does not discriminate between causes [C215], [C216], [C244]. Sample rate is excluded [C233]. Three candidates remain: a session level near 0.16 on the JBL endpoint alone [C222], the Harman Spatial APO registered on that endpoint [C231], [C235], and, if the tap sits after the mix to 8 channels, conversion of a 7.1 endpoint to stereo [C230], [C239]. Whether process loopback taps after session volume or after SFX is undocumented [C209], [C213], [C228].

**Q3.** The per-app setting routes a whole browser and not a tab, so replay needs its own binary [C301], [C311]. An issue reporter says the Settings page worked, in a report whose example is Chrome playing YouTube on a Windows 10 dev build [C306], yet the setting is reported as stored but unreliably applied at startup [C303], [C304], which makes a check at each start necessary. Nothing read suggests EME changes it, but no source tested that [C345]. setSinkId needs a permission and a user gesture [C319], [C321], Chrome lacks selectAudioOutput [C322], Spotify's fetched chunks do not call setSinkId [C325], [C326], and Chromium refuses a sink switch once a Web Audio client is attached, which Spotify may do [C334], [C336]. No code path read blocks setSinkId for EME, which is absence of evidence and not a test [C343]. One user reports the Windows assignment overriding setSinkId, so the two routes should not be combined without a test [C313], [C314].

## Findings

### Q1 — does process loopback capture real audio from a browser playing Widevine-protected Spotify, or is it silenced? — answered with a gap

- **C101** [quoted] On desktop Widevine has no audio decoder, so the browser decodes the audio itself after the CDM's Decrypt call — "we conclude that Widevine Desktop does not provide a built-in audio decoder, and therefore, Widevine leaves audio decoding to the caller, which is the browser" — [S1], section 3.2.1 Audio Content
- **C102** [quoted] In the paper's experiments audio goes through Decrypt only, which succeeds only with keys at the lowest security level — "calls to Decrypt only succeed if the license keys are configured at the lowest security level, namely SW_SECURE_CRYPTO" — [S1], section 3.2.1
- **C103** [quoted] The paper states audio is recovered by observing the decryption interface, because decrypted samples are returned before decoding — "we show that audio content can be trivially recovered because the decrypted samples are returned prior to decoding" — [S2], abstract
- **C104** [quoted] The paper describes Widevine as cross-browser across Firefox, Edge, Chrome and other Chromium browsers and as a software-only module — "cross-browser (Firefox, Edge, Chrome, and all other Chromium-based browsers). This portability comes with a price: Widevine deploys a software-only module" — [S1], section 1 Introduction
- **C105** [quoted] On the two premium services tested (Netflix, Disney+) audio tracks were not encrypted at all. Spotify is not in the paper — "In both platforms, we observe that audio tracks are not encrypted." — [S1], evaluation section, premium content providers
- **C106** [quoted] The paper places capture at the point of playback outside the DRM threat model, and gives recording with a microphone or scraping pixels as examples, but it does not test OS audio loopback — "this type of DRM bypassing is not taken into consideration in the DRM threat model" — [S1], Analog Hole discussion
- **C107** [quoted] A Mozilla engineer, reviewing a MediaElementSource test in 2017 for Firefox, states the policy for EME capture is to block video and not audio — "We don't care about restricting access to audio; we only want to block access to video." — [S3], comment 15 (Chris Pearce), bug RESOLVED FIXED in Firefox 55
- **C108** [quoted] The same bug reports Chrome returning a Web Audio source node for an EME element, and it was filed to match that in Firefox — "In Chrome, createMediaElementSource would return" — [S3], comment 0
- **C109** [quoted] Chromium does refuse captureStream on an EME element, for audio too, so one in-page route is blocked by the browser itself — "Stream capture not supported with EME" — [S4], around line 333, comment "Avoid capturing from EME-protected Media Elements."
- **C110** [quoted] A Firefox tab recorder add-on says EME-protected playback including the Spotify web player cannot be captured and that Firefox gives a silent stream. Its listed methods are in-page (media element, stream URL, Web Audio), not OS loopback — "DRM/EME-protected playback (Netflix, Spotify web player, and similar) cannot be captured." — [S5], AMO API JSON description (addons.mozilla.org/api/v5/addons/addon/tab-audio-rec/), version 0.1.4
- **C111** [inference] The in-page refusals are a browser-API layer above the OS mixer, so they say nothing about what the Windows audio engine receives. C110 is evidence on captureStream-style capture and not on WASAPI — from [C109], [C110], [C114]
- **C112** [quoted] COUNTER-CLAIM. Microsoft says a trusted audio driver does not let a loopback device capture digital streams holding protected content. The page's context is Vista DRM and hardware loopback devices, and process loopback is not named — "a trusted audio driver does not permit a loopback device to capture digital streams that contain protected content" — [S6], paragraph beginning "Windows Vista provides digital rights management"
- **C113** [quoted] Microsoft's protected path for audio is the Media Foundation PMP, where the audio renderer is a trusted output — "The EVR and the audio renderer both implement" — [S7], PMP walkthrough, the sentence continues "IMFTrustedOutput"
- **C114** [quoted] Without a hardware loopback pin, WASAPI loopback is a copy of the audio engine's output stream — "WASAPI copies the output stream from the audio engine into the loopback application" — [S6], paragraph on loopback implementation
- **C115** [quoted] The driver-level rule is stated for any digital-audio content protected by DRM, on a Windows driver page — "can be played only by trusted audio drivers" — [S8], intro, ms.date 2017-04-20
- **C116** [measured] Chromium's two Windows WASAPI render sources contain no reference to protect, drm, widevine or encrypted — Own measurements, M1
- **C117** [inference] Microsoft's rule (C112) is not shown to reach Widevine browser audio, and not shown to stop short of it. Its text speaks of DRM-protected content, trusted drivers and Media Foundation's protected path (C113, C115) and names no browser key system, while Widevine hands back samples the browser decodes (C101) and Chromium's render path sets no protection marker in the files read (C116). This is reasoning from code and docs, not an observation of the audio engine, and it covers Chromium only — from [C101], [C112], [C113], [C115], [C116]
- **C118** [quoted] Process loopback captures the named process and its children only — "only audio from the specified process, and its children, will be captured" — [S9], sample README
- **C119** [quoted] If the target process has no audio rendering streams, the capturing process receives silence — "process receives silence" — [S9], sample README, sentence begins "If the processes whose audio will be captured does not have any audio rendering streams"
- **C120** [quoted] One tool's bug: a missing stream flag made all process-loopback streams silent, for reasons unrelated to DRM — "causing all process-loopback streams to fail silently with no audio output" — [S10], release notes
- **C121** [quoted] COUNTER-CLAIM. The same tool later abandoned per-process loopback for system loopback, reporting silent failure on sandboxed and Electron apps, and says browsers then worked. It gives no cause and no mention of DRM — "failed silently for sandboxed and Electron-based apps" — [S11], release notes
- **C122** [quoted] OBS's application audio guide says some applications' audio may be incompatible with the source, with no mention of DRM, and it names Chrome and Spotify only for executable matching — "Due to how some applications output some of their audio" — [S12], application audio guide
- **C123** [quoted] One OBS forum report says application audio capture of Edge failed on 29.1.3, with no site named, no DRM mention and no resolution — "app audio capture of microsoft edge does not work" — [S13], original post
- **C124** [quoted] One Audacity report (2020) on Spotify in Chrome: WASAPI loopback failed to open the device with an error, and DirectSound with the Primary Sound Capture Driver recorded silence. So no loopback capture of the stream ran, and the thread has no answer — "recording does happen but only a silent recording" — [S14], post 1, 2020-04-09
- **C125** [quoted] A FreeBSD guide covers Widevine in Chromium, naming Spotify, and setting up a loopback device for browser audio. It is a different OS and not WASAPI, and it states no Spotify capture result — "listen to Spotify" — [S15], post text
- **C126** [snippet] Low-grade source: a Discord troubleshooting article says viewers hearing audio while Netflix video is black means audio path capture works and the protected video surface does not. Uncited SEO content — "When viewers hear audio but see no video" — [S16], search snippet
- **C128** [quoted] Spotify's Web Playback SDK requires iframes to allow encrypted-media. The page names no key system and concerns the SDK, not the consumer web player — "The Web Playback SDK require iframes to allow encrypted-media" — [S18], requirements list
- **C129** [quoted] Spotify lists Chrome, Firefox, Safari and Edge as supported — "supported by most web browsers (Chrome, Firefox, Safari" — [S18], requirements list
- **C130** [quoted] A Spotify community user ties web player failures to the Widevine component. This is an anecdote, but it is the only link found from the consumer web player to Widevine — "In this case your client (browser) is not able to get a license from widevine, this is why the songs are skipped." — [S19], comment 7008287 in the thread
- **C131** [inference] No source read names any output protection on EME audio on desktop: Widevine leaves audio to the browser, in the paper's experiments after a decrypt-only step at the lowest security level, and a Mozilla engineer, in a MediaElementSource review, states the capture restriction targets video and not audio. Chromium's captureStream refusal (C109) is the one audio-affecting limit found, and it is an in-page API rule, not output protection. No Widevine or W3C statement on the point was read — from [C101], [C102], [C103], [C107], [C109]
- **C132** [inference] No block mechanism was identified for process-loopback capture of a software Widevine browser stream; real audio is plausible from architecture but no direct report or test stands behind it. Silence seen in practice has at least two non-DRM causes (C119, C120), and Microsoft's protected-content rule stays an unresolved counter-claim for Edge or any Media Foundation path — from [C101], [C107], [C112], [C116], [C117], [C119], [C120], [C121]
- **C133** [prior] Chrome's hardware-secure and Edge's PlayReady playback use a Media Foundation renderer that may carry audio too, which would sit on the path C112 and C113 describe. Which key system the Spotify web player uses on Edge was not established — not fetched in this run
- **C134** [prior] Chromium renders audio in a utility process under the browser process, and Firefox uses a separate audio arrangement, so the capture root must be the top browser process for child streams to be included — not fetched in this run

### Q2 — why can process loopback read 16 dB lower on one endpoint when endpoint mute and volume have no effect? — answered with a gap

- **C201** [quoted] By default the loopback stream taps the audio before any volume or muting, as stated in the page's AEC reference loopback stream section — "By default, the loopback stream "taps into" (listens to) the audio stream prior to any volume or muting being applied." — [S20], Reference Loopback Stream
- **C202** [quoted] The win32 API states the same default and names the opt-in — "The default behavior is for the loopback stream to be tapped before volume and/or mute." — [S21], AUDCLNT_STREAMOPTIONS_POST_VOLUME_LOOPBACK
- **C203** [quoted] For the driver-side loopback pin, the default tap is before volume and mute on the mixer output — "If no attributes are applied to the loopback stream, then loopback should be collected before volume and mute is applied on the mixer output." — [S22], Remarks
- **C204** [quoted] Microsoft's summarized flow on the driver page lists volume as the next step in the flow — "Volume: The next step in the flow, where the volume of the audio signal is adjusted." — [S22], Remarks, flow list in order: signal processing output, volume, mute, EFX, output
- **C205** [quoted] Without a hardware loopback pin, WASAPI copies the engine output stream for the loopback client — "WASAPI copies the output stream from the audio engine into the loopback application's capture buffer" — [S6], implementation paragraph
- **C206** [quoted] Counter to C201: a third party measured device loopback tracking a software endpoint volume on a virtual device — "Diagnose mode on "Music (Elgato Virtual Audio)" showed the captured level ≈ meter peak + 20·log10(volume scalar)." — [S23], PR body, test plan
- **C207** [quoted] Hardware-engine loopback returns the final output; Microsoft ties that to AEC needing the final stream — "which requires knowledge of the final output stream to cancel echoes" — [S24], The hardware audio engine
- **C208** [quoted] Desktop apps never use the offload path, so a hardware-engine route does not apply to a desktop tone player — "Desktop applications cannot use the offloading capabilities of audio adapters" — [S24], Application Impact
- **C209** [quoted] Microsoft documents process loopback only as selecting the render streams of a process tree; the reference text read says nothing on volume, APOs or mix position — "The ID of the process for which the render streams, and the render streams of its child processes, will be included or excluded when activating the process loopback stream." — [S25], TargetProcessId
- **C210** [quoted] A third party measured process-loopback level as unchanged by endpoint mute, endpoint volume and endpoint sample rate — "the captured level is identical with the output muted, at 10 % and at 100 %, and with the output's own format at 44.1, 48 and 96 kHz" — [S26], PR body, testing
- **C211** [quoted] The same author states the tap is before device volume, mute and format — "taken **before the output device's volume, mute and format** are applied" — [S26], PR body
- **C212** [quoted] A third party measured process loopback as not hearing an Equalizer APO cut that device loopback did hear; the PR does not say which APO stage the cut ran in — "Measured with a -20 dB cut at 1 kHz in Equalizer APO: endpoint loopback hears the cut, process loopback doesn't" — [S27], PR body
- **C213** [quoted] Whether process loopback is before or after session mute was still an unmeasured question in the one thread found — "whether process loopback captures before or after the session mute" — [S28], issue body
- **C215** [inference] Process loopback sits upstream of endpoint volume and mute, resting on one third-party measurement plus the documented device-loopback default; no source read establishes its position against session volume or SFX — from [C201], [C209], [C210], [C212], [C213]
- **C216** [inference] The observation that endpoint mute and volume have no effect fits the documented default and does not discriminate between the remaining causes — from [C201], [C210]
- **C217** [quoted] Session volume is applied per stream by multiplying PCM samples by a linear effective level — "multiplies the PCM samples in the stream by the effective volume level" — [S29], page body
- **C218** [quoted] The docs give the decibel mapping for session levels — "setting v = 0.5 attenuates the loudness of the original signal (the signal before the volume level is applied) by 6 decibels" — [S29], page body
- **C219** [quoted] Render session volume and mute persist across restarts, so a stale per-app setting can survive — "the volume and mute settings for a rendering session are persistent across system restarts" — [S30], page body
- **C220** [quoted] Sndvol shows session sliders per rendering device, so a session setting can differ between two endpoints — "displays volume sliders for the audio sessions that are playing on each audio rendering device" — [S29], page body
- **C221** [measured] The gap computed from the stated levels 0.0353 and 0.0056 is 15.99 dB, factor 0.1586; the 0.0353 has no retained log — Own measurements, M5
- **C222** [inference] A session level near 0.16 on the JBL endpoint alone would produce the whole gap, but only if process loopback taps after session volume, which C213 leaves open — from [C213], [C217], [C218], [C220], [C221]
- **C223** [quoted] SFX runs per stream before the mix and may change channel count — "Stream effects are before the mix (render) or after the tee (capture) for a given mode and can be used for changing channel count before the mixer." — [S31], Stream Effect (SFX)
- **C224** [quoted] MFX runs after the mix for a given mode — "Mode effects are applied after the mix (render) or before the tee (capture) for a given mode" — [S31], Mode Effect (MFX)
- **C225** [quoted] EFX is always applied, raw streams included — "An endpoint effect is always applied, even to raw streams." — [S31], Endpoint Effect (EFX)
- **C226** [quoted] Raw streams skip SFX, which gives a test that removes the JBL SFX from the path — "Stream effects are not used for raw streams." — [S31], Stream Effect (SFX)
- **C227** [quoted] Order in the software engine: SFX per stream, then mix, then EFX — "can be applied on a per-stream basis before the separate streams are mixed, and then passed through any available endpoint effects (EFX)" — [S24], The software audio engine
- **C228** [inference] No doc read places session volume relative to SFX, MFX or the loopback tap — from [C204], [C217], [C227]
- **C229** [measured] The capture path in this repo is process loopback at a caller-fixed 44.1 kHz stereo 16-bit — Own measurements, M2
- **C230** [measured] JBL Quantum 800 Game is an 8-channel 7.1 endpoint at 48 kHz, while Realtek Digital Output is 2-channel at 48 kHz — Own measurements, M3
- **C231** [measured] JBL Game has a Harman Spatial APO registered as SFX and as pre-mix LFX, and Realtek Digital Output has no SFX, MFX or EFX CLSID among the keys read — Own measurements, M3, M4
- **C232** [measured] JBL Game is a physical USB endpoint (VID_0ECB PID_203E) with JBL driver 2.0.0.14 from oem1.inf, section QuantumAudio.Game.SysFx, not a virtual device; QuantumServer is running and the APO lives in System32\JBLQuantumENGINE\apo — Own measurements, M4
- **C233** [inference] Sample-rate conversion cannot explain the gap, because both endpoints run at 48 kHz, and C210 reports no level change across rates — from [C210], [C230]
- **C234** [inference] Among the registry keys read, the two endpoints differ in channel layout and the Spatial APO; session volume is a third candidate not visible there — from [C230], [C231], [C220]
- **C235** [inference] The Spatial APO can matter only if process loopback taps after SFX; C212 is the only evidence on that, it points against an APO effect but its stage is unknown, so SFX stays open — from [C212], [C223], [C231]
- **C236** [quoted] Microsoft's sample opens the process-loopback client at 44.1 kHz stereo 16-bit with the auto-convert flag — "AUDCLNT_STREAMFLAGS_AUTOCONVERTPCM" — [S32], lines 98-106
- **C237** [quoted] A third party reports some Windows headsets playing no sound through WebRTC's audio module when the shared-mode mix format has more than two channels; it gives no Windows coefficients — "some headsets play no sound: those whose shared-mode mix format has more than two channels" — [S33], PR body
- **C238** [prior] Windows does not publish the coefficients for its shared-mode channel matrix, and a matrix that normalises 8 channels into 2 can attenuate front channels by several dB; the size on this path is unverified — not fetched in this run
- **C239** [inference] If the tap sits after the stream is mixed to the endpoint's 8 channels, which no source read places, a 7.1 stream converted to stereo for the capture client is a live candidate for part of the gap, untestable from docs alone — from [C230], [C236], [C238]
- **C240** [snippet] A search summary of JBL support says TalkThrough in QuantumENGINE can make game audio very quiet — "check if the TalkThrough is activated via the QuantumENGINE software" — [S34], search summary; the only JBL article in that result, and curl of this URL returned no such text, so the attribution is unconfirmed
- **C241** [snippet] The same summary points to the Game-Chat balance dial for low game volume — "Check that your headset's Game-Chat balance dial and ensure that the balance is correct" — [S34], search summary; attribution unconfirmed as in C240
- **C242** [snippet] The manual lists QuantumSURROUND and DTS Headphone:X as Windows-only software features — "QuantumSURROUND and DTS Headphone:X V2.0 available on Windows only" — [S35], search summary
- **C243** [inference] No source says whether the balance dial or TalkThrough acts in the APO or in the headset; if in the headset, they sit downstream of any loopback tap and cannot change the capture — from [C232], [C240], [C241]
- **C244** [measured] On this machine, with a Python tone source on the default endpoint (JBL Game), capture RMS held at 0.00561 under endpoint mute, mute with volume 0, and volume 0.41, and byte counts were 352800 or 354564 in both muted and unmuted states, apart from a shorter first window (347508 bytes, RMS 0.00556, unmuted). No browser or DRM player was tested, and volume 0 without mute was not tested — Own measurements, M13
- **C245** [measured] The same tone read RMS 0.00561 on JBL Game, which is below the repo's RMS_SILENCE_THRESHOLD of 0.01, while the stated 0.0353 on an unused endpoint, which has no retained log, is above it, so on those values this gap decides whether the silence gate passes the capture — Own measurements, M13

### Q3 — do the Windows per-app output setting and setSinkId route such a tab? — answered with a gap

- **C301** [measured] Windows keeps per-app audio entries that pair an endpoint ID with the full image path of the executable, so Chrome and Edge are separate entries and no entry names a tab — Own measurements, M7
- **C302** [measured] Chrome has one unversioned path in the store here, while the Store build of Spotify has 28 distinct version-specific paths — Own measurements, M7
- **C303** [quoted] EarTrumpet's developers state the per-app setting is persisted but not reliably applied at startup, and they re-apply it themselves — "This setting is persisted however it is not reliably applied at startup. EarTrumpet mitigates this for the user by manually setting the persisted playback device." — [S36], Persisted output devices
- **C304** [quoted] A 2020 user report with Chrome: Windows resets the setting so that it still lists the device for Chrome while audio does not use it — "It does this in a way that for chrome the output is still listed to my 'digital' driver, but that it isnt being picked up on." — [S37], question post 2020-07-11, single report, no resolution shown
- **C305** [quoted] A second report, about a game not a browser: the choice holds while Windows runs and reverts on reboot — "it resets after rebooting the pc, it seems it stays the same as long as the pc is running." — [S38], question post
- **C306** [quoted] An issue reporter says the Windows Settings page works for moving an app's output, in a report whose example is Chrome playing YouTube on a Windows 10 dev build, while a third-party tool using the undocumented call failed — "No way to change it other than the regular windows interface Settings/System/Sound/App volume and device preferences (which works)." — [S39], issue body, 2021
- **C307** [quoted] EarTrumpet's maintainer names Chrome's process model as a complication for moving its audio — "Chrome is a bit of a bad test case because of the sandboxed tab processes, brokered audio, and other complexities." — [S39], comment by riverar 2021-06-07
- **C308** [quoted] EarTrumpet's maintainer confirms IAudioPolicyConfigFactory changed on Windows dev builds — "Can confirm, IAudioPolicyConfigFactory was changed." — [S39], comment by riverar 2021-06-13
- **C309** [quoted] Chromium's audio service runs in its own process on Windows — "In a separate process on Windows, Mac (sandboxed) and Linux (unsandboxed);" — [S40], opening list
- **C310** [quoted] For the default device Chromium opens the endpoint through GetDefaultAudioEndpoint; ids that are neither default nor loopback go to GetDevice — "device_enum->GetDefaultAudioEndpoint(data_flow, role, &endpoint_device);" — [S41], CreateDeviceInternal, with device_enum->GetDevice in the else branch
- **C311** [inference] The per-app setting routes a whole browser, not a tab: every Chrome tab shares one executable path and one audio service process, so two tabs cannot go to two endpoints this way, and a separate browser binary is what gives separation — from [C301], [C309]
- **C312** [inference] Chrome shows one path in the store here, so the path does not look like the persistence risk; the reported risk is the entry not being applied at startup — from [C302], [C303], [C304]
- **C313** [quoted] A user reports the Windows per-app assignment overriding setSinkId in Chrome: the OS-assigned device wins — "It seems, if you have Google Chrome assigned to a specific device via windows, the OS will overwrite it and still play audio through the OS assigned device." — [S42], comment by KyleGuay 2021-08-24, one user, not tested here
- **C314** [inference] The two routes may conflict and should not be combined without a test: the code opens an explicit device for a non-default sink, yet a user reports the Windows assignment wins — from [C310], [C313]
- **C315** [quoted] Firefox's audio library opens the default endpoint on Windows through the same system call — "hr = enumerator->GetDefaultAudioEndpoint(direction, role, device.receive());" — [S43], get_default_endpoint
- **C316** [snippet] A Firefox user found Firefox's per-app output stuck on an empty value after an issue with sound, fixed by setting it to Default — "there was the output of mozilla firefox set to "blank" instead of "Default". You cannot set this to blank with the gui yourself" — [S44], search snippet; the page renders by script and curl and WebFetch got no text, exact URL among the SUMO hits not confirmed
- **C317** [snippet] A Mozilla support answer says the newer Windows panel behaves differently from the classic Volume Mixer for Firefox entries — "the 1709/1803 update(s) to Windows 10 apparently created a new preferences panel that works differently from the classic Volume Mixer" — [S45], search snippet; page not readable by curl or WebFetch
- **C318** [quoted] A third-party guide attributes to Mozilla that a device change in the Volume Mixer may not apply correctly to an open tab, with a tab reload as the fix — "Mozilla notes that selecting another sound device in the Volume Mixer may not apply correctly to an existing tab. Reload the page after the change" — [S46], section "Reload after changing routing", low-grade source, original Mozilla page not found
- **C319** [quoted] setSinkId needs user permission for a non-default device — "User permission is required to access a non-default device." — [S47], Security requirements
- **C320** [quoted] The spec gates setSinkId behind the speaker-selection permissions policy — "If document is not allowed to use the feature identified by "speaker-selection" , return a promise rejected with a new DOMException whose name is NotAllowedError ." — [S48], setSinkId algorithm, text with tags stripped
- **C321** [quoted] selectAudioOutput requires transient activation, so it cannot run silently without a user gesture — "Transient user activation" — [S49], Security requirements, line reads: Transient user activation is required
- **C322** [measured] setSinkId ships in Chrome 49, Edge 17, Firefox 116 and Safari 18.4; selectAudioOutput ships in Firefox 116 but not in Chrome per the compat data — Own measurements, M8
- **C323** [quoted] In Chrome a non-default device goes through a browser-side permission check, and a default-device request skips it — "// Check device permissions if nondefault device is requested." — [S50], followed by a MediaDevicesPermissionChecker call with MediaDeviceType::kMediaAudioOutput
- **C324** [quoted] In practice a Chrome extension needed the site's microphone permission to set a sink, per the AudioPick project — "requires media (microphone) permissions to be granted to every site with audio sinks that need to be manipulated" — [S51], Note paragraph
- **C325** [measured] The Spotify web player chunks fetched contain no setSinkId, sinkId, selectAudioOutput or audiooutput string, while they do contain Widevine and PlayReady strings; lazy chunks were not fetched — Own measurements, M9
- **C326** [inference] The fetched chunks show no sign of calling setSinkId; lazy chunks are unread; if that holds, this route needs code injected into the page — from [C325]
- **C327** [quoted] AudioPick states that it reaches media objects outside the document tree, and says Spotify should work, in hedged wording, with no test reported — "e. g. **Spotify** and **SoundCloud** should now work with the extension, too." — [S51], intro; latest release v0.3.10 of 2024-01-25
- **C328** [quoted] A March 2026 Chrome Web Store review says AudioPick no longer works, cause unstated — "Not working anymore." — [S52], review by Eloi Lortie 2026-03-05, Extpose listing of AudioPick reviews; anecdotal
- **C329** [quoted] An older AudioPick report: in Chrome 88 selecting the default gave silence and any other device played through the system default — "if i select the system default device i hear nothing, if i select any other device i hear sound via my system default device..." — [S42], comment by Dr4inder 2021-02-25, Chrome 88
- **C330** [quoted] Firefox enabled the setSinkId pref by default in Firefox 116 — "Enable by default setSinkId pref" — [S53], bug title, RESOLVED FIXED 2023-06-29
- **C331** [quoted] An unconfirmed Firefox bug on YouTube Music: the sink held for the first track, then audio returned to default though sinkId still reported the chosen device — "After you change the music the output device is to default." — [S54], comment 0, 2026-09-26, Firefox 156, UNCONFIRMED; a second comment says it fails on YouTube entirely
- **C332** [quoted] An open Firefox bug: setSinkId() on a media element with a MediaStream srcObject can hang until playback ends when the device cannot be opened — "setSinkId() on a media element with MediaStream srcObject hangs until playback ends, when device cannot be opened" — [S55], title, status NEW
- **C333** [quoted] Chromium's SetSinkId forwards the sink ID to the audio source provider with no test for encryption or a CDM — "CrossThreadBindOnce(&SetSinkIdOnMediaThread, audio_source_provider_," — [S56], WebMediaPlayerImpl::SetSinkId
- **C334** [quoted] Chromium refuses a sink switch with an internal error once the element has a Web Audio client attached — "if (client_ || !sink_)" — [S57], SwitchOutputDevice, next line runs callback with OUTPUT_DEVICE_STATUS_ERROR_INTERNAL
- **C335** [measured] Spotify's player factory creates its media element detached with document.createElement and conditionally connects it to createMediaElementSource through an AudioContext of filters; the condition was not observed live — Own measurements, M9
- **C336** [inference] setSinkId on Spotify's element can fail in Chromium whenever that audio processor is active; this rests on reading code, and whether the processor is on by default is unknown — from [C334], [C335]
- **C337** [quoted] Gecko's SetSinkId checks the permissions policy, then calls the decoder's SetSink, and the shown function has no MediaKeys or encryption branch — "RefPtr<SinkInfoPromise> p = mDecoder->SetSink(aInfo)->Then(" — [S58], HTMLMediaElement::SetSinkId
- **C338** [measured] The W3C spec for audio output does not mention encrypted media, so it defines no exception for EME — Own measurements, M11
- **C339** [measured] One Spotify key-system configuration (video-hw-all) asks for software-level audio robustness (SW_SECURE_CRYPTO) with hardware video; the other configurations were not recorded; the one chosen in a session was not observed — Own measurements, M9
- **C340** [prior] SW_SECURE_CRYPTO means software decryption with decoding by the browser, so audio returns to the ordinary output path — not fetched in this run
- **C341** [measured] Chromium's Media Foundation renderer file has no audio device or sink handling — Own measurements, M12
- **C342** [quoted] Chromium notes that with an external Renderer many methods won't work, giving GetCurrentFrameFromCompositor as the example; SetSinkId is not named — "// TODO(xhwang): When we use an external Renderer, many methods won't work," — [S56], near line 610
- **C343** [inference] No code path read blocks setSinkId because media is EME, and no report of such a failure was found, but this is absence of evidence: no run with Widevine playback was done — from [C333], [C337], [C338], [C344]
- **C344** [measured] Three GitHub searches for setSinkId with encrypted, DRM and widevine returned no issues — Own measurements, M10
- **C345** [inference] Nothing read suggests EME changes the per-app route, yet no source tested it with Widevine media — from [C306], [C310]

## Disconfirming evidence

- [C131], [C132] — query: `WASAPI loopback protected content DRM audio stream muted silent loopback capture blocked Windows audio engine` — found [C112] (Microsoft rule) as a counter-claim; a third-party itch.io recorder page warning about DRM audio being unavailable restates it without test.
- [C132] — query: `Windows ProcessLoopback capture DRM protected audio silent "protected" Apple Music OR Netflix app WASAPI recorder github issue no audio` — only the Microsoft page; no issue reporting a DRM block on process loopback.
- [C132] — query: `Spotify web player recording silent Chrome loopback capture blocked no sound recorded application audio not captured browser` — found [C110], an in-page recorder, not OS loopback.
- [C132] — query: `OBS "application audio capture" chrome.exe OR msedge.exe Spotify Netflix browser audio captured or silent forum github` — [C122], [C123]; nothing tying silence to DRM.
- [C132] — query: `record Spotify web player Chrome browser WASAPI loopback tool github per-app audio "web player" recording works` — [C121] only; no tool confirmed for the web player.
- [C131] — query: `Netflix black screen but audio works Discord stream OBS browser capture application audio DRM` — [C126] only; SEO pages agree audio survives but cite nothing.
- [C132] — query: `"Stereo Mix" OR "VB-Cable" OR "loopback" record Spotify web player in browser works Widevine audio not protected analog hole` — [C125] only; nothing on Windows.
- [C201] — query: `WASAPI loopback level changes with system master volume software volume device without hardware volume loopback attenuated by endpoint volume` — found against: [C206], plus old driver-dependent forum reports not cited. Device loopback only; [C210] covers process loopback.
- [C210] — query: `Windows process loopback ... capture level follows application volume mixer slider session volume ISimpleAudioVolume` — nothing against on endpoint volume; nothing settling session volume ([C213]).
- [C212] — query: `process loopback capture application audio includes APO effects SFX Equalizer APO spatial sound not applied, per-app volume mixer ignored or applied GitHub issue` — no opposite result; [C212] itself came from this query.
- [C239] — query: `process loopback stereo capture on 7.1 surround endpoint quieter downmix application loopback multichannel default device level drop spatial sound` — nothing relevant to Windows either way.
- [C222] — query: `OBS Application Audio Capture volume mixer app volume affects captured level process loopback muted in Windows mixer` — only a search summary of one unnamed guide; no source page, not a claim here.
- [C243] — query: `JBL QuantumENGINE JBL Quantum 800 Game Chat audio endpoints virtual surround DTS 7.1 Windows driver low volume recording loopback` — nothing on loopback; nothing locating the dial.
- [C306] — query: `Windows 11 "App volume and device preferences" Chrome output device resets ignored after restart browser audio goes to default device` — found [C304], [C305] and generic guides, not cited. No Chromium bug found.
- [C343] — query: `setSinkId encrypted media Widevine EME audio output device fails chromium "setSinkId" protected content`, plus the searches in M10 — nothing found against. A Shaka issue on Widevine output protection concerns display output, not audio sinks.
- [C326], [C327] — query: `AudioPick extension Spotify not working output device github issue chrome setSinkId DRM` — issue 15 of 2018, superseded, plus [C328], [C329]. No report ties a failure to DRM.
- [C316] — query: `Firefox ignores Windows app audio output device setting "firefox.exe" Volume mixer output device not working per-app`; Bugzilla quicksearch `App volume and device`, `windows per-app audio output device` and `wasapi default device windows sound settings app` returned 0 bugs — only [C316] to [C318] remain.

## Coverage

| Channel | Read | Gap |
|---|---|---|
| Users | [S13], [S14], [S37], [S39], [S42] | no report of process loopback on a Widevine stream, positive or negative |
| Specialists | [S6], [S40], [S3], [S43], [S29] | no Widevine or W3C statement on audio output protection; no official Microsoft documentation of the per-app setting |
| Data | [S1], [S26] | one third-party measurement of the process-loopback tap; no benchmark on Widevine audio |
| Adversarial | [S11], [S27], [S54] | disconfirming searches above found no DRM block report, which is absence of evidence |

Declared overruns of the ceiling: Q1 ran four searches and four reads, one read over, because the WOOT paper is the primary source on Widevine audio. Q2 ran ten searches (seven regular, three disconfirming): the first four reached neither process loopback nor the JBL endpoint. Q3 ran seven topic searches plus three disconfirming: early searches returned only generic text, so primary code and issue threads were reached by direct fetch.

Blocked or unread: issues.chromium.org gave no HTTP response, so the Chromium tracker was not searched. support.mozilla.org pages render by script, so those claims are snippets. extensionauditor.com returned 403. Spotify lazy chunks were not fetched. No Widevine playback was run in this pass.

## Own measurements

### M1 — Chromium WASAPI render sources, protection references

Chromium WASAPI render sources, protection references

Command: `for f in media/audio/win/audio_low_latency_output_win.cc media/audio/win/core_audio_util_win.cc; do curl -sL https://raw.githubusercontent.com/chromium/chromium/main/$f | grep -i -c -E "protect|drm|widevine|encrypted"; done`
Output: audio_low_latency_output_win.cc: 0; core_audio_util_win.cc: 0
Limit: absence in two files at main on 2026-11-02. Not a proof for the whole audio stack, and it says nothing about Firefox or about Edge's fork.

### M2 — Capture path in this repo

Grep of src/music_intel_mcp/_wasapi_loopback.py: "AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK" at line 9; line 98 "*, target_pid: int, sample_rate: int = 44100, channels: int = 2"; line 135 requires hdr_bits 16.

### M3 — Endpoint formats and effects, registry read

Command: scratch q2_endpoints.ps1, read-only registry under HKLM MMDevices\Audio\Render. Output trimmed:
  Realtek Digital Output / Realtek(R) Audio {1f97a042-...}: DeviceFormat tag=0xFFFE ch=2 sr=48000 bits=16 mask=0x3; no FX lines.
  Динамики / JBL Quantum 800 Game {65a9b767-...}: DeviceFormat tag=0xFFFE ch=8 sr=48000 bits=16 mask=0x63F; FX SFX and PreMixLFX {3D640EF5-6952-3BE3-9ED4-D2D20BC847B3} Spatial APO.

### M4 — JBL Game endpoint identity

Command: scratch q2_jbl.ps1, read-only. Output trimmed:
  CLSID InprocServer32 -> C:\Windows\System32\JBLQuantumENGINE\apo\SpatialAPO.dll. Copyright Harman International Industries.
  FxProperties {b725f130-47ef-101a-a5f1-02608c9eebac},10 = JBL 3D effect & Equalizer.
  Properties: oem1.inf:...:QuantumAudio.Game.SysFx:2.0.0.14:USB\VID_0ECB&PID_203E&MI_00.
  Driver: JBL Quantum 800 Game, provider JBL, oem1.inf, 2.0.0.14. Process: QuantumServer.
  Spatial-sound selection (Windows Sonic etc.) was not read.

### M5 — Size of the gap

Command: python -c 20*log10(0.0353/0.0056). Output: 15.991733567632442 and ratio 0.15864022662889518.

### M6 — Stored per-app volumes (failed, inconclusive)

failed, inconclusive (two attempts, the skill's limit). q2_jbl.ps1 and q2_sessions.ps1 looked for stored per-app volumes in HKCU Internet Explorer\LowRegistry\Audio\PolicyConfig\PropertyStore by endpoint GUID. Output: 1188 entries, 0 hits, because keys hold device-interface paths ("{2}.\\?\hdaudio#...") and not MMDevice GUIDs. Next attempt: match "vid_0ecb&pid_203e" and read values under subkey {219ED5A0-9CBF-4F3A-B927-37C9E5C5F14F} (GUID from memory, unverified). No audio was played; no tone test was run here, since it would sound in the user's headset.

### M7 — Per-app store on this machine

per-app store on this machine (Windows 11 26300). Read of
HKCU\Software\Microsoft\Internet Explorer\LowRegistry\Audio\PolicyConfig\PropertyStore, default value of each subkey,
GUIDs masked. Output: 1188 subkeys. Counts by value text: chrome.exe 5, msedge.exe 14, firefox.exe 0, Spotify.exe 51.
Distinct paths: chrome.exe 1, Spotify.exe 28. Value form:
{2}.\\?\bthenum#{GUID}_localmfg&0002#{GUID}\src/00010001|\Device\HarddiskVolume6\Program Files\Google\Chrome\Application\chrome.exe%b{GUID}
Spotify Store entries carry the version in the path (...SpotifyAB.SpotifyMusic_1.296.518.0_x64__...\Spotify.exe).
Not shown: which property each entry holds (volume, mute or device). Whether Firefox is installed or used here was not checked, so 0 says nothing about how Firefox treats the setting.

### M8 — Browser compatibility data for setSinkId and selectAudioOutput

browser-compat-data, raw JSON from github.com/mdn/browser-compat-data main, api/HTMLMediaElement.json and api/MediaDevices.json.
setSinkId version_added: chrome 49, edge 17, firefox 116, safari 18.4, firefox_android false, chrome_android false.
selectAudioOutput: chrome false (impl_url crbug.com/372214870), edge mirror, firefox 116, safari false.

### M9 — Spotify web player bundles

Spotify web player bundles. curl of https://open.spotify.com/ with a desktop UA, then the 7 script URLs in the HTML. Three large chunks
downloaded (574 KB, 1.58 MB, 4.53 MB), the locale file and manifest-web-player.1609946b.js came back empty, lazy chunks not fetched.
grep -F counts by files: setSinkId 0, selectAudioOutput 0, sinkId 0, audiooutput 0, requestMediaKeySystemAccess 1, com.widevine.alpha 2.
Occurrences: createMediaElementSource 2, com.microsoft.playready 5, SW_SECURE_CRYPTO 13, HW_SECURE 9.
Verbatim from chunk 3: createPlayer:e=>{let t=document.createElement(e);return r.isSupported()&&i&&r.connectAudioProcessor(t),t}
and connectAudioProcessor(e){let t=new AudioContext({latencyHint:"playback"}),r=t.createMediaElementSource(e);this.webAudioFilters=...
Verbatim from chunk 2: [{label:"video-hw-all",robustness:{audio:"SW_SECURE_CRYPTO",video:"HW_SECURE_ALL"}},...
What i and isSupported() evaluate to in a live session was not observed. No playback was run.

### M10 — GitHub issue search for setSinkId with EME terms

GitHub issue search: gh search issues "setSinkId encrypted", "setSinkId DRM", "setSinkId widevine". Output: empty for all three.

### M11 — W3C audio-output spec text

W3C spec text: https://www.w3.org/TR/audio-output/ fetched, grep -c -i for "encrypted media", "protected content", "DRM": 0 lines.

### M12 — Chromium and Gecko code on sink changes for EME or external renderers

Chromium media/renderers/win/media_foundation_renderer.cc (main, 72,895 bytes): grep -i for audio.*(device|endpoint|sink), SetSink: no match;
only SetVolume lines match the wider pattern. third_party/blink/renderer/platform/media/web_media_player_impl.cc SetSinkId body has no encrypted or CDM test.
Gecko dom/media/mediasink/AudioSink.cpp grep -i for encrypt, cdm: no real match (only "elementCount" substrings).

### M13 — Endpoint mute and volume against process-loopback capture

Run by hand on this machine, Python tone source on the default endpoint (JBL Game), one line per capture window, while the endpoint was muted and its volume changed by hand. The script only read mute and volume, at the end of each 2 s window, so a line next to a change may show the state after it.

Output, verbatim, run 20:47:35:

```
mute=0 vol=1.00 bytes=347508 rms=0.00556
mute=1 vol=1.00 bytes=352800 rms=0.00561
mute=1 vol=1.00 bytes=352800 rms=0.00561
mute=1 vol=1.00 bytes=352800 rms=0.00561
mute=1 vol=1.00 bytes=352800 rms=0.00561
mute=1 vol=1.00 bytes=352800 rms=0.00561
mute=1 vol=1.00 bytes=352800 rms=0.00561
mute=0 vol=1.00 bytes=352800 rms=0.00561
mute=1 vol=0.00 bytes=352800 rms=0.00561
mute=1 vol=0.00 bytes=354564 rms=0.00561
mute=1 vol=0.00 bytes=352800 rms=0.00561
mute=0 vol=0.41 bytes=352800 rms=0.00561
mute=0 vol=1.00 bytes=354564 rms=0.00561
mute=0 vol=1.00 bytes=352800 rms=0.00561
mute=0 vol=1.00 bytes=354564 rms=0.00561
mute=0 vol=1.00 bytes=354564 rms=0.00561
mute=0 vol=1.00 bytes=352800 rms=0.00561
mute=0 vol=1.00 bytes=352800 rms=0.00561
mute=0 vol=1.00 bytes=352800 rms=0.00561
mute=0 vol=1.00 bytes=352800 rms=0.00561
```

The 0.0353 figure for the unused endpoint is a result from earlier in the same session, from a routed run of the same tone. No file with that output survives, so it is a stated value and not a retained log.

Threshold, command: search of `src/music_intel_mcp/replay_capture.py`. Output: `54:RMS_SILENCE_THRESHOLD = 0.01`.

## Open questions

- Q1 — no new information — one bench run settles it: play a Spotify web track in the chosen browser, capture with process loopback rooted at the top browser PID, compare RMS and a fingerprint of the capture with the track. Repeat per browser for Edge and Firefox if those are used; the key system on Edge was not established.
- Q2 — ceiling — four tests on JBL Game separate the candidates: move the per-app slider while capturing, disable audio enhancements, open a RAW stream that skips SFX [C226], and set the endpoint to a 2-channel format. Each changes system audio settings, so the user runs them by hand.
- Q3 — blocked: no live run — set the browser to the target endpoint in the Volume mixer, play the web player, read the endpoint peak meter, then repeat after a restart of the browser and of Windows. Whether Spotify's connectAudioProcessor is on by default stays unobserved [C336], and Firefox behaviour rests on snippets [C316], [C317] and a low-grade guide [C318].
- Label verdict — open: no capture of a real Widevine stream exists, so the Q1 answer and Microsoft's unresolved protected-content rule stay unsettled, and the cause of the 16 dB gap is not isolated; the bench run under Q1 and the four tests under Q2 settle them.

## Sources

- [S1] www.usenix.org: woot26-roudot.pdf — https://www.usenix.org/system/files/woot26-roudot.pdf — passage
- [S2] www.usenix.org: roudot — https://www.usenix.org/conference/woot26/presentation/roudot — passage
- [S3] bugzilla.mozilla.org: show_bug.cgi?id=1331763 — https://bugzilla.mozilla.org/show_bug.cgi?id=1331763 — passage
- [S4] raw.githubusercontent.com: html_media_element_capture.cc — https://raw.githubusercontent.com/chromium/chromium/main/third_party/blink/renderer/modules/mediacapturefromelement/html_media_element_capture.cc — passage
- [S5] addons.mozilla.org: tab-audio-rec — https://addons.mozilla.org/firefox/addon/tab-audio-rec/ — passage
- [S6] learn.microsoft.com: loopback-recording — https://learn.microsoft.com/en-us/windows/win32/coreaudio/loopback-recording — passage
- [S7] learn.microsoft.com: protected-media-path — https://learn.microsoft.com/en-us/windows/win32/medfound/protected-media-path — passage
- [S8] learn.microsoft.com: digital-rights-management — https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/digital-rights-management — passage
- [S9] raw.githubusercontent.com: README.md — https://raw.githubusercontent.com/microsoft/Windows-classic-samples/main/Samples/ApplicationLoopback/README.md — passage
- [S10] github.com: v1.2.9 — https://github.com/trstanis/clownboard/releases/tag/v1.2.9 — passage
- [S11] github.com: v1.2.10 — https://github.com/trstanis/clownboard/releases/tag/v1.2.10 — passage
- [S12] obsproject.com: application-audio-capture-guide — https://obsproject.com/kb/application-audio-capture-guide — passage
- [S13] obsproject.com: app-audio-capture-of-microsoft-edge-does-not-work.169305 — https://obsproject.com/forum/threads/app-audio-capture-of-microsoft-edge-does-not-work.169305/ — passage
- [S14] forum.audacityteam.org: 56303 — https://forum.audacityteam.org/t/recording-streamed-audio-e-g-spotify-problem/56303 — passage
- [S15] forums.freebsd.org: post?id=659409 — https://forums.freebsd.org/goto/post?id=659409 — passage
- [S16] ponderworthy.com: why-you-cant-stream-netflix-on-discord-and-how-to-actually-fix-the-black-screen-1iw6 — https://ponderworthy.com/why-you-cant-stream-netflix-on-discord-and-how-to-actually-fix-the-black-screen-1iw6 — snippet only
- [S17] obsproject.com: 4480 — https://obsproject.com/forum/resources/win-capture-audio.1338/update/4480/ — snippet only
- [S18] developer.spotify.com: web-playback-sdk — https://developer.spotify.com/documentation/web-playback-sdk — passage
- [S19] community.spotify.com: true — https://community.spotify.com/t5/Desktop-Windows/Webplayer-cannot-play-content-stops-playing-after-10-seconds/m-p/7008331/highlight/true — passage
- [S20] learn.microsoft.com: windows-11-apis-for-audio-processing-objects — https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/windows-11-apis-for-audio-processing-objects — passage
- [S21] learn.microsoft.com: ne-audioclient-audclnt_streamoptions — https://learn.microsoft.com/en-us/windows/win32/api/audioclient/ne-audioclient-audclnt_streamoptions — passage
- [S22] learn.microsoft.com: ksproperty-audioloopback — https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/ksproperty-audioloopback — passage
- [S23] github.com: 2 — https://github.com/F3NN3X/InfoPanel.AudioMeter/pull/2 — passage
- [S24] learn.microsoft.com: hardware-offloaded-audio-processing — https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/hardware-offloaded-audio-processing — passage
- [S25] learn.microsoft.com: ns-audioclientactivationparams-audioclient_process_loopback_params — https://learn.microsoft.com/en-us/windows/win32/api/audioclientactivationparams/ns-audioclientactivationparams-audioclient_process_loopback_params — passage
- [S26] github.com: 3432 — https://github.com/alvr-org/ALVR/pull/3432 — passage
- [S27] github.com: 1 — https://github.com/rizqyep/leungo/pull/1 — passage
- [S28] github.com: 365 — https://github.com/Raaif-Yousuf/Wingman/issues/365 — passage
- [S29] learn.microsoft.com: session-volume-controls — https://learn.microsoft.com/en-us/windows/win32/coreaudio/session-volume-controls — passage
- [S30] learn.microsoft.com: audio-sessions — https://learn.microsoft.com/en-us/windows/win32/coreaudio/audio-sessions — passage
- [S31] learn.microsoft.com: audio-processing-object-architecture — https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/audio-processing-object-architecture — passage
- [S32] github.com: LoopbackCapture.cpp — https://github.com/microsoft/Windows-classic-samples/blob/main/Samples/ApplicationLoopback/cpp/LoopbackCapture.cpp — passage
- [S33] github.com: 358 — https://github.com/devopvoid/webrtc-java/pull/358 — passage
- [S34] support.jbl.com: 000016959.html — https://support.jbl.com/us/en/howto/jbl-quantum-microphone-set-up-teammates-can-t-hear-you-us/000016959.html — snippet only
- [S35] manuals.plus: jbl-quantum-800-manual — https://manuals.plus/jbl/jbl-quantum-800-manual — snippet only
- [S36] github.com: README.md — https://github.com/File-New-Project/EarTrumpet/blob/master/EarTrumpet/README.md — passage
- [S37] learn.microsoft.com: 5587745 — https://learn.microsoft.com/en-us/answers/a/5587745 — passage
- [S38] learn.microsoft.com: app-volume-and-device-preferences-does-not-remembe — https://learn.microsoft.com/en-us/answers/questions/4170305/app-volume-and-device-preferences-does-not-remembe — passage
- [S39] github.com: 835 — https://github.com/File-New-Project/EarTrumpet/issues/835 — passage
- [S40] chromium.googlesource.com: README.md — https://chromium.googlesource.com/chromium/src/+/main/services/audio/README.md — passage
- [S41] chromium.googlesource.com: core_audio_util_win.cc — https://chromium.googlesource.com/chromium/src/+/main/media/audio/win/core_audio_util_win.cc — passage
- [S42] github.com: 44 — https://github.com/rain-fighters/AudioPick/issues/44 — passage
- [S43] raw.githubusercontent.com: cubeb_wasapi.cpp — https://raw.githubusercontent.com/mozilla/cubeb/master/src/cubeb_wasapi.cpp — passage
- [S44] support.mozilla.org: 1323720 — https://support.mozilla.org/en-US/questions/1323720 — snippet only
- [S45] support.mozilla.org: 1232158 — https://support.mozilla.org/en-US/questions/1232158 — snippet only
- [S46] umatechnology.org: firefox-sound-not-working-on-windows-a-complete-troubleshooting-guide — https://umatechnology.org/firefox-sound-not-working-on-windows-a-complete-troubleshooting-guide/ — passage
- [S47] developer.mozilla.org: setSinkId — https://developer.mozilla.org/en-US/docs/Web/API/HTMLMediaElement/setSinkId — passage
- [S48] www.w3.org: audio-output — https://www.w3.org/TR/audio-output/ — passage
- [S49] developer.mozilla.org: selectAudioOutput — https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/selectAudioOutput — passage
- [S50] chromium.googlesource.com: audio_output_authorization_handler.cc — https://chromium.googlesource.com/chromium/src/+/main/content/browser/renderer_host/media/audio_output_authorization_handler.cc — passage
- [S51] github.com: README.md — https://github.com/rain-fighters/AudioPick/blob/main/README.md — passage
- [S52] extpose.com: en — https://extpose.com/ext/142/en — passage
- [S53] bugzilla.mozilla.org: show_bug.cgi?id=1498512 — https://bugzilla.mozilla.org/show_bug.cgi?id=1498512 — passage
- [S54] bugzilla.mozilla.org: show_bug.cgi?id=2075890 — https://bugzilla.mozilla.org/show_bug.cgi?id=2075890 — passage
- [S55] bugzilla.mozilla.org: show_bug.cgi?id=1874351 — https://bugzilla.mozilla.org/show_bug.cgi?id=1874351 — passage
- [S56] chromium.googlesource.com: web_media_player_impl.cc — https://chromium.googlesource.com/chromium/src/+/main/third_party/blink/renderer/platform/media/web_media_player_impl.cc — passage
- [S57] chromium.googlesource.com: web_audio_source_provider_impl.cc — https://chromium.googlesource.com/chromium/src/+/main/third_party/blink/renderer/platform/media/web_audio_source_provider_impl.cc — passage
- [S58] raw.githubusercontent.com: HTMLMediaElement.cpp — https://raw.githubusercontent.com/mozilla-firefox/firefox/main/dom/html/HTMLMediaElement.cpp — passage

## Audit

| Claim | Verdict | Note |
|---|---|---|
| C101 | SUPPORTED |  |
| C102 | PARTIAL | paper says "In our experiments, we only witness calls to the Decrypt function"; scoped to the experiments |
| C103 | SUPPORTED |  |
| C104 | PARTIAL | paper states the cross-browser design; the attack was evaluated on Firefox ESR 140 with Widevine 4.10.2931.0; "findings cover" narrowed |
| C105 | SUPPORTED |  |
| C106 | SUPPORTED |  |
| C107 | PARTIAL | the sentence is about a MediaElementSource test (next sentence: it can only capture audio); context added |
| C108 | SUPPORTED |  |
| C109 | SUPPORTED |  |
| C110 | SUPPORTED | page is script-built; passage read through the AMO API JSON; relabelled quoted |
| C111 | SUPPORTED |  |
| C112 | SUPPORTED |  |
| C113 | SUPPORTED |  |
| C114 | SUPPORTED |  |
| C115 | PARTIAL | page says "Any digital-audio content that is protected by DRM"; no "Windows DRM" wording and no browser or key system named; claim rewritten |
| C116 | SUPPORTED | rerun: 0 and 0 |
| C117 | PARTIAL | S6, S7, S8 name no Widevine or key system; the rule is neither shown to reach nor to stop short of browser audio; narrowed |
| C118 | SUPPORTED |  |
| C119 | PARTIAL | README says "receives silence"; "not an error" and the PID conclusion are not in it; narrowed |
| C120 | SUPPORTED |  |
| C121 | SUPPORTED |  |
| C122 | PARTIAL | guide says audio may be incompatible; claim aligned |
| C123 | SUPPORTED |  |
| C124 | SUPPORTED |  |
| C125 | SUPPORTED |  |
| C126 | UNCHECKABLE | S16 returns only a script redirect shell; no text obtained |
| C127 | UNSUPPORTED | S17 calls v2.0.0-beta a full rewrite on AUDIOCLIENT_PROCESS_LOOPBACK_PARAMS, "no longer injecting and hooking", so "hook-based" is wrong; removed |
| C128 | PARTIAL | page does not say web playback runs on EME for the consumer player; "so" dropped |
| C129 | SUPPORTED |  |
| C130 | SUPPORTED | drafted snippet quote is not on the page; page read from raw HTML, another passage in the thread supports the claim and replaces it; the thread also shows a widevine-license/v1/audio/license 400; relabelled quoted; quote script reports NOT FOUND because the comment text sits in JSON embedded in the HTML, found there by hand |
| C131 | PARTIAL | rests on narrowed C102, C107; scope carried over |
| C132 | PARTIAL | rests on narrowed C102, C107, C117; "real audio is expected" softened to plausible with no direct test |
| C133 | UNCHECKABLE | prior |
| C134 | UNCHECKABLE | prior |
| C201 | PARTIAL | page text is in the AEC reference loopback stream section; "on every render endpoint" and the opt-in are not in the quote; narrowed |
| C202 | SUPPORTED |  |
| C203 | SUPPORTED |  |
| C204 | PARTIAL | quote shows only the volume step; order moved to the place field as read from the list |
| C205 | SUPPORTED |  |
| C206 | SUPPORTED |  |
| C207 | SUPPORTED |  |
| C208 | SUPPORTED |  |
| C209 | SUPPORTED |  |
| C210 | SUPPORTED |  |
| C211 | SUPPORTED |  |
| C212 | SUPPORTED |  |
| C213 | SUPPORTED |  |
| C215 | SUPPORTED |  |
| C216 | SUPPORTED |  |
| C217 | SUPPORTED |  |
| C218 | SUPPORTED |  |
| C219 | SUPPORTED |  |
| C220 | SUPPORTED |  |
| C221 | PARTIAL | M5 uses 0.0056 where M13 reads 0.00561 (15.98 dB); 0.0353 has no retained log; limit stated |
| C222 | SUPPORTED |  |
| C223 | SUPPORTED |  |
| C224 | SUPPORTED |  |
| C225 | PARTIAL | "after the mix" and "every stream" are not in the quote; narrowed |
| C226 | SUPPORTED |  |
| C227 | PARTIAL | "then hardware" is not in the quote; dropped |
| C228 | SUPPORTED |  |
| C229 | SUPPORTED | lines 9, 98, 125, 135 of the repo file match |
| C230 | SUPPORTED |  |
| C231 | SUPPORTED |  |
| C232 | SUPPORTED |  |
| C233 | SUPPORTED |  |
| C234 | PARTIAL | "exactly two" goes past the keys read; narrowed |
| C235 | SUPPORTED |  |
| C236 | PARTIAL | the conversion clause is not in the quote; format at lines 98-100 confirmed; dropped |
| C237 | PARTIAL | source concerns WebRTC's audio module, not Windows audio code in general; narrowed |
| C238 | UNCHECKABLE | prior |
| C239 | PARTIAL | follows only if the tap sits after the mix to 8 channels; rests on prior C238; made conditional |
| C240 | UNCHECKABLE | S34 returned 403; no text obtained |
| C241 | UNCHECKABLE | S34 returned 403; no text obtained |
| C242 | UNCHECKABLE | S35 returned 403; no text obtained |
| C243 | SUPPORTED | bases C240, C241 are uncheckable; the claim is stated conditionally and holds as a statement of absence |
| C244 | PARTIAL | byte counts took a third value in the first window; recorded output restated |
| C245 | PARTIAL | 0.0353 is a stated value with no log; condition added |
| C301 | SUPPORTED |  |
| C302 | PARTIAL | M7 shows 1 and 28 distinct paths; orphaning is not measured; dropped |
| C303 | SUPPORTED |  |
| C304 | PARTIAL | post says Windows resets after reboot; "until reselected" sits outside the quote; rewritten to the quote |
| C305 | SUPPORTED | quote is in a follow-up reply dated 2023-03-26, not in the question post |
| C306 | PARTIAL | quote is the reporter's generic statement; example of Chrome playing YouTube comes from the same issue; attribution tightened |
| C307 | SUPPORTED |  |
| C308 | PARTIAL | quote confirms the change only; "broke until patched" dropped |
| C309 | SUPPORTED |  |
| C310 | PARTIAL | quote script reports NOT FOUND on the HTML view (unfetchable on another run); quote and else branch read in plain source via ?format=TEXT; branch condition corrected |
| C311 | SUPPORTED |  |
| C312 | PARTIAL | follows only as far as one path seen here; narrowed |
| C313 | SUPPORTED | date 2021-08-24 confirmed |
| C314 | SUPPORTED |  |
| C315 | PARTIAL | "which is where a per-app preference would have to act" is unsourced; dropped |
| C316 | UNCHECKABLE | S44 is a script shell; the API routes return the same 3038-byte shell |
| C317 | UNCHECKABLE | S45 is a script shell; the API routes return the same 3038-byte shell |
| C318 | SUPPORTED | guide read in full; passage set as quote; relabelled quoted; still a low-grade source |
| C319 | PARTIAL | secure context and prompt are not in the quote; narrowed |
| C320 | SUPPORTED |  |
| C321 | SUPPORTED |  |
| C322 | SUPPORTED |  |
| C323 | SUPPORTED | read via ?format=TEXT: default device returns early, then the quoted comment and the permission check on kMediaAudioOutput |
| C324 | SUPPORTED |  |
| C325 | SUPPORTED |  |
| C326 | PARTIAL | conclusion made conditional on the unread lazy chunks |
| C327 | SUPPORTED | outside-tree statement sits in the same sentence; release v0.3.10 of 2024-01-25 confirmed by gh |
| C328 | SUPPORTED | review read on the Extpose page; relabelled quoted; same list has later positive reviews (2026-06-09, 2026-06-19) and "Broken on audible" (2026-05-13) |
| C329 | PARTIAL | rewritten to the single report quoted; "no effect" was wider |
| C330 | PARTIAL | bug has no add-on content; add-on part dropped; date confirmed by Bugzilla REST |
| C331 | SUPPORTED | comment 0 confirmed; the same comment also has the sinkId-still-reported part |
| C332 | PARTIAL | status NEW confirmed; "not the EME path" is the author's gloss; dropped |
| C333 | SUPPORTED | quote script reports NOT FOUND on the HTML view; found at line 1320 of the plain source via ?format=TEXT |
| C334 | SUPPORTED | quote script reports NOT FOUND on the HTML view; found in SwitchOutputDevice of the plain source via ?format=TEXT |
| C335 | SUPPORTED |  |
| C336 | PARTIAL | AudioContext.setSinkId clause has no base; dropped |
| C337 | SUPPORTED | S58 URL is 404, the file moved; read at dom/media/mediaelement/HTMLMediaElement.cpp; policy check, mDecoder->SetSink and no MediaKeys or encrypt in lines 8782-8860 confirmed |
| C338 | SUPPORTED | re-grep for encrypt, EME, DRM, protected content: 0 |
| C339 | PARTIAL | M9 records one configuration verbatim; plural dropped |
| C340 | UNCHECKABLE | prior |
| C341 | PARTIAL | "its Windows hardware-secure path" is not in M12; dropped |
| C342 | PARTIAL | source does not name SetSinkId or a sink; "leaves open" reduced to what the comment says |
| C343 | SUPPORTED |  |
| C344 | SUPPORTED | rerun: all three empty |
| C345 | PARTIAL | rests on C306, C310, neither of which says the route "acts below the browser"; dropped |

Label verdict: open — no direct test of process loopback on Widevine audio exists, Microsoft's protected-content rule stays unresolved and could change the Q1 answer, and the cause of the 16 dB gap is not isolated.
Quote check: replay-process-loopback-drm-2026-11-02.md: 75 quoted claim(s): 62 found, 4 NOT FOUND, 9 unfetchable, 0 no source

## Disposition

| Recommendation | State | Where |
|---|---|---|
| Keep the process-loopback route for #262, and make its first step one bench capture on a real Spotify web-player tab | deferred | Osasuwu/music-intel-mcp#262 |
| Route replay through a separate browser binary on the Windows per-app setting, verify that route at every start | deferred | Osasuwu/music-intel-mcp#262 |
| Do not build on setSinkId | deferred | Osasuwu/music-intel-mcp#262 |
| Explain the 16 dB gap before the #169 gate runs on real captures | deferred | Osasuwu/music-intel-mcp#262 |
