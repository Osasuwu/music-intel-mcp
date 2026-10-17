"""The one fake GitHub client the `.github/scripts` tests drive their callers through (#2068).

`FakeApi` stands in for `github_client.Client` at the caller's seam: the scripts
take the client as a parameter (`make_client`), so their tests route GETs and
writes here instead of monkeypatching a per-module request helper.
`FakeTransport` sits one layer lower, under a real `Client`, for the tests of the
client's own request, retry and download policy.
"""

import importlib.util
import io
import json
import sys
import urllib.error
from pathlib import Path
from types import SimpleNamespace

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".github" / "scripts").is_dir())
CLIENT_MODULE = "_jarvis_github_client"


def load_client():
    """`.github/scripts/github_client.py`, under the sys.modules key the scripts use,
    so `ListingTruncated` is one class across the scripts and the tests."""
    if CLIENT_MODULE in sys.modules:
        return sys.modules[CLIENT_MODULE]
    spec = importlib.util.spec_from_file_location(
        CLIENT_MODULE, _ROOT / ".github" / "scripts" / "github_client.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[CLIENT_MODULE] = module
    spec.loader.exec_module(module)
    return module


_UNROUTED = object()


def _route(routes, method, path, body):
    """The answer of the longest prefix of `path` in `routes`, or `_UNROUTED`.

    Longest prefix wins, so 'pulls/7/files' is never answered as 'pulls/7'. A
    callable value is called with the request and its result answered; an
    exception instance (given or called up) is raised."""
    for prefix in sorted(routes, key=len, reverse=True):
        if path.startswith(prefix):
            value = routes[prefix]
            if callable(value):
                value = value(method, path, body)
            if isinstance(value, BaseException):
                raise value
            return value
    return _UNROUTED


class FakeApi:
    """GETs answered from `responses`, other methods from `writes`, both by the
    longest matching path prefix; downloads by artifact id from `blobs`.

    An unrouted GET is a test bug and raises AssertionError; an unrouted write
    answers `{}`. Every call is recorded in `calls` as (method, path, body), and
    each one made with `idempotent=False` also in `once` as (method, path)."""

    def __init__(self, responses=None, blobs=None, writes=None):
        self.responses = responses or {}
        self.blobs = blobs or {}
        self.writes = writes or {}
        self.calls = []
        self.once = []

    def __call__(self, method, path, body=None, *, idempotent=True):
        self.calls.append((method, path, body))
        if not idempotent:
            self.once.append((method, path))
        routes = self.responses if method == "GET" else self.writes
        answer = _route(routes, method, path, body)
        if answer is not _UNROUTED:
            return answer
        if method != "GET":
            return {}
        raise AssertionError(f"unexpected API call {method} {path}")

    def download(self, path, read):
        """The blob of artifact `.../artifacts/<id>/zip`, unread: `read` is the
        client's job, and the blob already stands for what it returns. `read` is
        still required, as `Client.download` requires it."""
        self.calls.append(("DOWNLOAD", path, None))
        blob = self.blobs[int(path.split("/")[-2])]
        if isinstance(blob, BaseException):
            raise blob
        return blob


class SequencedApi(FakeApi):
    """Answers each read of PR #7 with the next of `prs`; the file list is empty."""

    def __init__(self, *prs):
        super().__init__()
        self.prs = iter(prs)

    def __call__(self, method, path, body=None, *, idempotent=True):
        self.calls.append((method, path, body))
        if not idempotent:
            self.once.append((method, path))
        if method != "GET":
            return {}
        return next(self.prs) if path == "repos/o/r/pulls/7" else []


# --- under a real Client --------------------------------------------------------


class FakeResponse:
    """What a transport returns: a context manager with `read` and `headers`."""

    def __init__(self, body=b"", headers=None, status=200):
        self._body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.headers = headers or {}
        self.status = status

    def read(self, amt=None):
        return self._body if amt is None else self._body[:amt]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def http_error(code, body=b"", headers=None, url="https://api.github.com/x"):
    """An HTTPError as urllib raises it, with a readable body."""
    data = body.encode() if isinstance(body, str) else body
    return urllib.error.HTTPError(url, code, "error", headers or {}, io.BytesIO(data))


class FakeTransport:
    """The `transport(request, timeout, redirects)` a real `Client` is built with.

    `script` is a list of outcomes answered in order, or a dict of URL -> list of
    outcomes, each URL's answered in order. An outcome is an exception (raised), a
    `FakeResponse`, or a body (bytes, or JSON-able). Every request is recorded in
    `requests` with its url, method, lower-cased headers, body, timeout and
    redirects."""

    def __init__(self, script):
        self.script = script
        self.requests = []

    def __call__(self, request, timeout, redirects):
        url = request.full_url
        self.requests.append(
            SimpleNamespace(
                url=url,
                method=request.get_method(),
                headers={k.lower(): v for k, v in request.header_items()},
                body=json.loads(request.data) if request.data else None,
                timeout=timeout,
                redirects=redirects,
            )
        )
        queue = self.script if isinstance(self.script, list) else self.script.get(url)
        if not queue:
            raise AssertionError(f"no scripted answer for {request.get_method()} {url}")
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome if isinstance(outcome, FakeResponse) else FakeResponse(outcome)
