"""PreToolUse hook: scan tool inputs for secret patterns before execution.

Handles three tool shapes:
- File writes (Edit/Write/NotebookEdit): scans the text being written to disk
- Bash: scans the command string for literal secrets and exfiltration patterns
- Everything else (GitHub MCP writes, and any tool with text fields): scans the
  common text-carrying fields

Reads the PreToolUse payload from stdin (JSON). Exits 2 to block when a secret
is detected, 0 otherwise. Scans only for credentials that grant access — not
for personal data.

Fail-open on unparseable input is deliberate: a hook that blocks on malformed
payloads would wedge the session, and the CI gitleaks job is the second layer
that catches whatever a broken hook lets past.
"""

import json
import re
import sys

# ---------------------------------------------------------------------------
# Patterns that indicate real secrets (API keys, tokens, passwords).
# Tuned for a low false-positive rate — only high-confidence shapes.
# ---------------------------------------------------------------------------
SECRET_PATTERNS = [
    # AWS
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
    # Anthropic
    (r"sk-ant-[a-zA-Z0-9_-]{20,}", "Anthropic API Key"),
    # GitHub tokens
    (r"gh[ps]_[A-Za-z0-9_]{36,}", "GitHub Token"),
    (r"github_pat_[A-Za-z0-9_]{22,}", "GitHub PAT"),
    # JWT-shaped bearer tokens (eyJ... base64)
    (r"eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{10,}", "JWT"),
    # OpenAI-style
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style API Key"),
    # Slack
    (r"xox[bpras]-[A-Za-z0-9-]{10,}", "Slack Token"),
    # Private keys
    (r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----", "Private Key"),
    # Generic credential assignment — catches the provider keys this project
    # actually holds (Spotify client secret, Last.fm / AcoustID keys, Discogs
    # token), whose own value shapes are too generic to match on their own.
    (
        r"""(?i)(?:password|secret|token|api_key|apikey)\s*[:=]\s*['"]?[A-Za-z0-9_/+.-]{16,}""",
        "Credential assignment",
    ),
]

COMPILED_SECRETS = [(re.compile(p), label) for p, label in SECRET_PATTERNS]

# ---------------------------------------------------------------------------
# Bash-specific: command shapes that exfiltrate secrets. These match the
# COMMAND STRING, not the secret values themselves.
# ---------------------------------------------------------------------------

# Env var names holding this project's live credentials. Their *values* should
# never be expanded into a command that leaves the machine.
_SECRET_VARS = (
    r"SPOTIFY_CLIENT_SECRET|LASTFM_API_KEY|ACOUSTID_API_KEY"
    r"|DISCOGS_TOKEN|GITHUB_TOKEN"
)

BASH_DANGER_PATTERNS = [
    # Reading .env and piping/redirecting somewhere. `.env.example` holds
    # placeholders, not values, and is excluded so ordinary setup work is not
    # blocked by a guard aimed at the real file.
    (r"(?:cat|type|Get-Content|gc)\s+[^\|;]*\.env(?!\.example)", "Reading .env file"),
    # Expanding a secret env var inside an HTTP command
    (
        rf"(?:curl|wget|http|Invoke-WebRequest|iwr)\s.*\$(?:{_SECRET_VARS})",
        "Secret var in HTTP command",
    ),
    (
        rf"(?:curl|wget|http|Invoke-WebRequest|iwr)\s.*\$\{{(?:{_SECRET_VARS})\}}",
        "Secret var in HTTP command",
    ),
    # Piping the environment to a network tool
    (r"(?:env|printenv|set)\s*\|.*(?:curl|wget|nc|ncat|socat|http)", "Env dump to network"),
    # Sending .env contents via curl -d / --data
    (r"curl\s.*(?:-d|--data)\s*@?\.env(?!\.example)", "Sending .env via curl"),
    # netcat/socat with .env
    (r"(?:nc|ncat|socat)\s.*\.env(?!\.example)", "Sending .env via netcat"),
    # base64 encoding .env (obfuscation attempt)
    (r"base64\s.*\.env(?!\.example)", "Encoding .env"),
    (r"\.env(?!\.example).*\|\s*base64", "Encoding .env"),
]

COMPILED_BASH = [(re.compile(p, re.IGNORECASE), label) for p, label in BASH_DANGER_PATTERNS]


# ---------------------------------------------------------------------------
# Text extraction per tool shape
# ---------------------------------------------------------------------------

# Fields carrying text destined for disk, per file-write tool:
#   Write        -> content
#   Edit         -> new_string
#   NotebookEdit -> new_source
# Only the *incoming* text is scanned. `old_string` is deliberately excluded:
# it is text already on disk, and blocking there would make an existing secret
# unremovable by the one tool that could remove it.
FILE_WRITE_TOOLS = ("Edit", "Write", "NotebookEdit")
_FILE_WRITE_KEYS = ("content", "new_string", "new_source")

_TEXT_FIELD_KEYS = ("body", "title", "content", "message", "description", "comment")

# This scanner's own test file exists to hold secret-SHAPED fixtures; scanning
# it blocks the one file that must contain them. Exempt by basename only —
# deliberately not a `tests/` prefix, because a test directory is exactly where
# a real leaked key would otherwise hide.
SELF_TEST_BASENAMES = ("test_hooks_secret_scanner.py",)


def is_self_test_fixture(tool_input: dict) -> bool:
    """True when the write targets this scanner's own fixture file."""
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if not isinstance(path, str):
        return False
    basename = path.replace("\\", "/").rsplit("/", 1)[-1]
    return basename in SELF_TEST_BASENAMES


def extract_file_write_text(tool_input: dict) -> str:
    """Pull the text a file-write tool is about to put on disk."""
    parts = []
    for key in _FILE_WRITE_KEYS:
        val = tool_input.get(key)
        if isinstance(val, str):
            parts.append(val)
    # Batched edits: list of {old_string, new_string}
    edits = tool_input.get("edits")
    if isinstance(edits, list):
        for e in edits:
            if isinstance(e, dict) and isinstance(e.get("new_string"), str):
                parts.append(e["new_string"])
    return "\n".join(parts)


def extract_text_fields(tool_input: dict) -> str:
    """Pull common text-carrying fields from any other tool's input."""
    parts = []
    for key in _TEXT_FIELD_KEYS:
        val = tool_input.get(key)
        if isinstance(val, str):
            parts.append(val)
    # push_files-style batches: list of {path, content}
    files = tool_input.get("files")
    if isinstance(files, list):
        for f in files:
            if isinstance(f, dict) and isinstance(f.get("content"), str):
                parts.append(f["content"])
    return "\n".join(parts)


# Heredoc bodies: <<'EOF' ... EOF (multiline). The closing delimiter line ends
# the match at a `)` (subshell close), a following newline (more commands after
# the heredoc — the common case), or end-of-string.
_HEREDOC_RE = re.compile(
    r"<<-?\s*'?(\w+)'?\s*\n.*?\n\s*\1\s*(?:\)|(?=\n)|$)",
    re.DOTALL,
)


def strip_heredocs(command: str) -> str:
    """Remove heredoc bodies from a command string.

    Heredoc content is text, not executable command — only the executable part
    is checked against BASH_DANGER_PATTERNS. scan_secrets still sees the FULL
    command, because a literal key is dangerous wherever it appears.
    """
    return _HEREDOC_RE.sub("", command)


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


def scan_secrets(text: str) -> list[str]:
    """Check text for literal secret values."""
    return [label for pattern, label in COMPILED_SECRETS if pattern.search(text)]


def scan_bash_dangers(command: str) -> list[str]:
    """Check a bash command for exfiltration shapes, heredoc bodies excluded."""
    executable_part = strip_heredocs(command)
    return [label for pattern, label in COMPILED_BASH if pattern.search(executable_part)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def block(findings: list[str]) -> None:
    """Emit the deny decision and exit 2."""
    types = ", ".join(sorted(set(findings)))
    result = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": (
                f"BLOCKED: secret pattern detected ({types}). Remove credentials before retrying."
            ),
        }
    }
    json.dump(result, sys.stdout)
    sys.exit(2)


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        sys.exit(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        sys.exit(0)  # can't parse — don't wedge the session

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    if not isinstance(tool_input, dict):
        sys.exit(0)

    findings = []

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if not isinstance(command, str) or not command:
            sys.exit(0)
        findings.extend(scan_secrets(command))
        findings.extend(scan_bash_dangers(command))
    elif tool_name in FILE_WRITE_TOOLS:
        if is_self_test_fixture(tool_input):
            sys.exit(0)
        text = extract_file_write_text(tool_input)
        if not text:
            sys.exit(0)
        findings.extend(scan_secrets(text))
    else:
        text = extract_text_fields(tool_input)
        if not text:
            sys.exit(0)
        findings.extend(scan_secrets(text))

    if findings:
        block(findings)

    sys.exit(0)


if __name__ == "__main__":
    main()
