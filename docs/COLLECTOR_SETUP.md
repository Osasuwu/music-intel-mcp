# Running the audio collector

Thanks for helping test this. This app listens to whatever's playing in
Spotify / YouTube Music / SoundCloud on your PC and saves a small numeric
"fingerprint" (embedding) of each track locally on your machine. **The audio
itself is never saved or sent anywhere** — only the fingerprint, and only to
your own computer. Nothing leaves your PC unless you choose to send the
resulting file back.

## Requirements

- Windows 10/11.
- [Python 3.11 or newer](https://www.python.org/downloads/) (during install,
  check "Add python.exe to PATH").
- [Build Tools for Visual Studio 2022](https://visualstudio.microsoft.com/downloads/)
  — under "Tools for Visual Studio". You only need the **"Desktop development
  with C++"** workload, not the full Visual Studio IDE. This is needed once,
  to compile a small helper that talks to Windows' audio system.
- Git (to get the code) — [git-scm.com](https://git-scm.com/downloads).

## 1. Get the code

```powershell
git clone https://github.com/Osasuwu/music-intel-mcp.git
cd music-intel-mcp
```

## 2. Run the setup script

```powershell
.\scripts\setup_collector.ps1
```

This creates a private Python environment, installs everything needed, and
builds the audio-capture helper. At the end it'll tell you about one manual
step (below) before you can start listening.

## 3. Get the two model files (manual — one-time)

The app uses two pretrained models to recognize tracks. They're free to use
but their license doesn't allow this project to redistribute them, so you
grab them yourself from the people who publish them:

1. Go to <https://essentia.upf.edu/models.html>.
2. Download the **`discogs-effnet-bsdynamic`** model (the `.onnx` file and
   its matching `.json` file).
3. Download the **`mtg_jamendo_top50tags-discogs-effnet`** model (again both
   the `.onnx` and `.json` files).
4. Put all four files into the `.scratch\models\` folder inside the repo you
   cloned (the setup script prints the exact path and re-checks for you if
   you re-run it).

## 4. Start the app

```powershell
.venv\Scripts\python.exe -m music_intel_mcp.desktop_app
```

A small icon appears in your system tray (bottom-right, near the clock). It
shows what it last captured, and has a **Quit** option to stop. Just leave it
running in the background while you listen to music normally — Spotify,
YouTube Music in a browser tab, or SoundCloud in a browser tab all work,
since it watches Windows' own "now playing" info rather than any one app.

## What gets saved, and where

Everything stays in `data\` inside the repo folder (or wherever the
`MUSIC_INTEL_DATA_DIR` environment variable points, if you've set one) —
nothing is uploaded automatically. If you're asked to share your results,
you'll do that explicitly (e.g. zipping that folder and sending it over).

## Troubleshooting

- **Tray icon shows "error on '...'" repeatedly for the same app** — that
  app's audio session might not have started yet; it usually recovers on the
  next track.
- **Nothing ever gets captured** — check that the two model files are
  actually named exactly as listed above and sit directly in
  `.scratch\models\`, not a subfolder.
- **Setup script fails at the "Building the WASAPI native helper" step** —
  double check the C++ Build Tools installed the "Desktop development with
  C++" workload specifically, not just the base installer.
