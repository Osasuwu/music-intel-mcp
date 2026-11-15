"""setup_collector.ps1 must skip the native helper build when the exe exists (#278).

The script drives pip, venv and downloads, so it is not run here; the test
pins the gate in its text: the build call sits in the ``else`` branch of a
``Test-Path`` on the exact exe path the collector loads.
"""

import re
from pathlib import Path

from music_intel_mcp import _wasapi_loopback

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup_collector.ps1"


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8").replace("\r\n", "\n")


def test_build_runs_only_when_helper_exe_absent():
    text = _text()
    match = re.search(
        r"if \(Test-Path \$helperExe\) \{(?P<skip>.*?)\} else \{(?P<build>.*?)\n\}",
        text,
        re.DOTALL,
    )
    assert match, "build must be gated by Test-Path $helperExe with an else branch"
    assert "build_wasapi_helper.ps1" not in match["skip"]
    assert "skipping" in match["skip"]
    assert "build_wasapi_helper.ps1" in match["build"]
    assert text.count("build_wasapi_helper.ps1") == 1


def test_helper_path_matches_the_one_the_collector_loads():
    m = re.search(r'\$helperExe = Join-Path \$repoRoot "([^"]+)"', _text())
    assert m
    resolved = (SCRIPT.parents[1] / m.group(1).replace("\\", "/")).resolve()
    assert resolved == _wasapi_loopback._HELPER_PATH.resolve()
