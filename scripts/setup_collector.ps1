# Setup for the local audio-collection desktop app (#136, #139).
#
# Gets a fresh checkout from "git clone" to "double-click the tray icon":
# venv, live-capture + desktop extras, the native WASAPI helper build, the
# fpcalc chromaprint binary (#139 AC7), and instructions for the two ONNX
# model files (which this script deliberately does NOT download -- see the
# printed notice below).
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $repoRoot ".venv"
$modelsDir = Join-Path $repoRoot ".scratch\models"
$toolsDir = Join-Path $repoRoot ".scratch\tools"
$fpcalcExe = Join-Path $toolsDir "fpcalc.exe"
$chromaprintVersion = "1.5.1"
$chromaprintZipUrl = "https://github.com/acoustid/chromaprint/releases/download/v$chromaprintVersion/chromaprint-fpcalc-$chromaprintVersion-windows-x86_64.zip"

Write-Host "== music-intel collector setup ==" -ForegroundColor Cyan

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtualenv at $venvDir"
    python -m venv $venvDir
}

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "venv python not found at $venvPython -- venv creation failed?"
}

Write-Host "Installing music-intel-mcp with live-capture + desktop extras..."
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -e "$repoRoot[live-capture,desktop]"

Write-Host "Building the WASAPI native helper..."
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "build_wasapi_helper.ps1")

if (Get-Command fpcalc -ErrorAction SilentlyContinue) {
    Write-Host "fpcalc already on PATH -- skipping chromaprint fetch."
} elseif (Test-Path $fpcalcExe) {
    Write-Host "fpcalc already fetched at $fpcalcExe -- add $toolsDir to PATH if needed."
} else {
    Write-Host "Fetching fpcalc (chromaprint $chromaprintVersion) for AcoustID fingerprinting (#139 AC7)..."
    New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
    $zipPath = Join-Path $toolsDir "chromaprint-fpcalc.zip"
    Invoke-WebRequest -Uri $chromaprintZipUrl -OutFile $zipPath
    Expand-Archive -Path $zipPath -DestinationPath $toolsDir -Force
    Remove-Item $zipPath
    $extracted = Get-ChildItem -Path $toolsDir -Filter "fpcalc.exe" -Recurse | Select-Object -First 1
    if ($null -eq $extracted) {
        throw "fpcalc.exe not found after extracting $chromaprintZipUrl"
    }
    if ($extracted.FullName -ne $fpcalcExe) {
        Move-Item -Path $extracted.FullName -Destination $fpcalcExe -Force
    }
    Write-Host "fpcalc fetched to $fpcalcExe -- add $toolsDir to PATH (or pass fpcalc_path explicitly)."
}

if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
}

$requiredFiles = @(
    "discogs-effnet-bsdynamic-1.onnx",
    "discogs-effnet-bsdynamic-1.json",
    "mtg_jamendo_top50tags-discogs-effnet-1.onnx",
    "mtg_jamendo_top50tags-discogs-effnet-1.json"
)
$missing = $requiredFiles | Where-Object { -not (Test-Path (Join-Path $modelsDir $_)) }

Write-Host ""
Write-Host "== One manual step left: the ONNX models ==" -ForegroundColor Yellow
Write-Host "These are Essentia's Discogs-EffNet + MTG-Jamendo models, licensed"
Write-Host "CC BY-NC-ND 4.0 (decision 29852699) -- this project never bundles or"
Write-Host "redistributes them, so you fetch them yourself from the official source:"
Write-Host "  https://essentia.upf.edu/models.html"
Write-Host "  (look for 'discogs-effnet-bsdynamic' and 'mtg_jamendo_top50tags')"
Write-Host ""
Write-Host "Place these four files in:"
Write-Host "  $modelsDir"
Write-Host "  - discogs-effnet-bsdynamic-1.onnx  (+ its .json sidecar)"
Write-Host "  - mtg_jamendo_top50tags-discogs-effnet-1.onnx  (+ its .json sidecar)"
Write-Host ""

if ($missing.Count -eq 0) {
    Write-Host "All four files are already present -- you're done." -ForegroundColor Green
    Write-Host ""
    Write-Host "Start the app:"
    Write-Host "  $venvPython -m music_intel_mcp.desktop_app"
} else {
    Write-Host "Still missing:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  - $_" }
    Write-Host ""
    Write-Host "Once they're in place, start the app:"
    Write-Host "  $venvPython -m music_intel_mcp.desktop_app"
}
