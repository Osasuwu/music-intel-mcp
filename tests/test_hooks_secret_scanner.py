"""The PreToolUse secret guard must fail closed.

`.claude/hooks/secret-scanner.py` is invoked here exactly as Claude Code
invokes a project hook: a subprocess with the PreToolUse JSON payload on
stdin, no repo package context. These tests fail if the guard stops blocking,
and if `.claude/settings.json` stops wiring it — a hook that exists but is not
wired to a matcher protects nothing.

Fixture credentials are built by concatenation rather than as contiguous
literals, so authoring this file does not itself trip the write-time scanner.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "secret-scanner.py"
SETTINGS_PATH = REPO_ROOT / ".claude" / "settings.json"

_FAKE_AWS_KEY = "AKIA" + "0123456789ABCDEF"
_FAKE_ANTHROPIC_KEY = "sk-ant-" + ("a" * 25)
_FAKE_PROVIDER_SECRET = "b" * 32


def _run_hook(payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_hook_file_exists():
    assert HOOK_PATH.exists(), ".claude/hooks/secret-scanner.py must exist"


def test_hook_is_wired_on_every_intended_surface():
    """The scanner must be reachable from file writes, Bash, and GitHub writes."""
    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    entries = settings["hooks"]["PreToolUse"]

    wired = {
        entry["matcher"]
        for entry in entries
        if any("secret-scanner.py" in h.get("command", "") for h in entry["hooks"])
    }

    assert any("Write" in m for m in wired), "file-write tools are unguarded"
    assert any(m == "Bash" for m in wired), "Bash is unguarded"
    assert any("mcp__github__" in m for m in wired), "GitHub MCP writes are unguarded"


def test_blocks_bash_command_carrying_a_key():
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "export AWS_KEY=" + _FAKE_AWS_KEY},
    }
    result = _run_hook(payload)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "hookSpecificOutput" in result.stdout


def test_blocks_file_write_carrying_a_key():
    payload = {
        "tool_name": "Write",
        "tool_input": {
            "file_path": "notes.txt",
            "content": "key value is " + _FAKE_ANTHROPIC_KEY,
        },
    }
    result = _run_hook(payload)
    assert result.returncode == 2, result.stdout + result.stderr


def test_blocks_provider_credential_assignment():
    """Provider keys have no distinctive value shape — the assignment is the tell."""
    payload = {
        "tool_name": "Write",
        "tool_input": {
            "file_path": "config.py",
            "content": "SPOTIFY_CLIENT_SECRET=" + _FAKE_PROVIDER_SECRET,
        },
    }
    result = _run_hook(payload)
    assert result.returncode == 2, result.stdout + result.stderr


def test_blocks_issue_body_carrying_a_key():
    """The leak path this guards is a value pasted into an issue or PR body."""
    payload = {
        "tool_name": "mcp__github__issue_write",
        "tool_input": {
            "title": "Investigate auth failure",
            "body": "the token that fails is " + _FAKE_ANTHROPIC_KEY,
        },
    }
    result = _run_hook(payload)
    assert result.returncode == 2, result.stdout + result.stderr


def test_blocks_env_file_exfiltration():
    payload = {
        "tool_name": "Bash",
        "tool_input": {"command": "cat .env | base64"},
    }
    result = _run_hook(payload)
    assert result.returncode == 2, result.stdout + result.stderr


def test_allows_reading_the_env_example():
    """`.env.example` holds placeholders; blocking it would break ordinary setup."""
    payload = {"tool_name": "Bash", "tool_input": {"command": "cat .env.example"}}
    result = _run_hook(payload)
    assert result.returncode == 0, result.stdout + result.stderr


def test_allows_benign_bash_command():
    payload = {"tool_name": "Bash", "tool_input": {"command": "ls -la"}}
    result = _run_hook(payload)
    assert result.returncode == 0, result.stdout + result.stderr


def test_allows_benign_file_write():
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": "notes.txt", "content": "hello world"},
    }
    result = _run_hook(payload)
    assert result.returncode == 0, result.stdout + result.stderr
