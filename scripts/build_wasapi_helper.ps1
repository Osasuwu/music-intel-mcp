# Builds native/wasapi_loopback_helper/main.cpp -> wasapi_loopback_helper.exe
# using the MSVC toolchain (VS2022 Community, x64 host/target).
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$srcDir = Join-Path $repoRoot "native\wasapi_loopback_helper"
$vcvars = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

if (-not (Test-Path $vcvars)) {
    throw "vcvars64.bat not found at $vcvars - is VS2022 Community installed?"
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
