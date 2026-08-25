# Builds native/wasapi_loopback_helper/main.cpp -> wasapi_loopback_helper.exe
# using the MSVC toolchain (VS2022, x64 host/target). Full VS2022 Community is
# not required -- the much smaller "Build Tools for Visual Studio 2022" (just
# the "Desktop development with C++" workload) is enough, which is the path
# friend-setup steers people toward (#136).
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$srcDir = Join-Path $repoRoot "native\wasapi_loopback_helper"

$vcvarsCandidates = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
$vcvars = $vcvarsCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $vcvars) {
    throw "vcvars64.bat not found in any known VS2022 install location. Install " + `
        "'Build Tools for Visual Studio 2022' (https://visualstudio.microsoft.com/downloads/, " + `
        "under 'Tools for Visual Studio') with the 'Desktop development with C++' workload, " + `
        "then re-run this script."
}

Push-Location $srcDir
try {
    $cmd = '"' + $vcvars + '" && cl.exe /nologo /EHsc /std:c++17 /O2 main.cpp /Fe:wasapi_loopback_helper.exe /link Mmdevapi.lib Ole32.lib'
    cmd /c $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }
    Write-Host "Built: $srcDir\wasapi_loopback_helper.exe"
} finally {
    Pop-Location
}
