"""The one GitHub REST client for the gate scripts in `.github/scripts` (#2068).

Every caller takes a `Client` as a parameter instead of building its own
request, so one request shape, one retry policy and one capped paginator serve
them all, and their tests drive one fake (`tests/ci/_github_fake.py`).

Retry policy, the same for every method (a request refused by a rate limit was
never processed, and an idempotent write converges on a retry):

- 502/503/504: wait `Retry-After` (capped at 60 s), else `min(2**attempt, 60)`;
  a call made with `idempotent=False` raises at once instead, since a 5xx can
  follow a committed write and a retry would write it twice.
- 403/429: `Retry-After` (capped at 60 s); else an exhausted primary limit
  (`x-ratelimit-remaining: 0`) waits for its reset, 1-60 s, or 60 s with no
  reset; else a 429, or a body naming the secondary rate limit, waits
  `min(2**attempt, 60)`. Any other 403 is a real permission failure and raises.
- Anything else (3xx included), URLError and timeouts raise at once.

Attempts are 1-indexed and capped at `MAX_ATTEMPTS`, so a persistent 5xx sleeps
2, 4, 8 and 16 s before its fifth and last try. A final error other than a 404
(an answer some callers expect) prints `METHOD path -> HTTP code: <body>` to
stderr, the body on one line, so a rejected token shows its reason (#2057).

Stdlib only. Callers load this file by path, beside them; it reads no
environment, so the token is always the caller's choice.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API_ROOT = "https://api.github.com"
TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
MAX_ATTEMPTS = 5
MAX_DELAY = 60
RETRIED_5XX = (502, 503, 504)
THROTTLED = (403, 429)
REDIRECTS = (301, 302, 303, 307, 308)
SECONDARY_LIMIT_HINT = "secondary rate limit"


class ListingTruncated(RuntimeError):
    """A listing the verdict depends on did not end within its page cap (#2076).

    `items` holds what was read; it is not the whole list."""

    def __init__(self, path, max_pages, per_page, items):
        super().__init__(
            f"listing {path} truncated: its last allowed page ({max_pages} of "
            f"{per_page}) was full, so it may go on unread"
        )
        self.items = items


def paged(api, path, key=None, per_page=100, max_pages=10):
    """All items of a paginated list endpoint.

    A listing whose last allowed page is still full raises `ListingTruncated`
    rather than be handed on as whole: what it leaves unread could be the one
    item that decides the verdict. A page past the cap is not probed, since an
    endpoint that stops serving at a result limit (the runs API, 1000) would read
    empty there and look complete.

    A page that is not a list (or holds none under `key`) raises ValueError: read
    as an empty page it would end the listing early and pass for the whole list."""
    joiner = "&" if "?" in path else "?"
    items = []
    for page in range(1, max_pages + 1):
        data = api("GET", f"{path}{joiner}per_page={per_page}&page={page}")
        if key:
            batch = data.get(key) if isinstance(data, dict) else None
        else:
            batch = data
        if not isinstance(batch, list):
            where = f" under {key!r}" if key else ""
            raise ValueError(
                f"listing {path} page {page}: expected a list{where}, got {type(batch).__name__}"
            )
        items.extend(batch)
        if len(batch) < per_page:
            return items
    raise ListingTruncated(path, max_pages, per_page, items)


def remove_label(api, repo, number, name):
    """Remove label `name` from issue or PR `number`: True when it was removed,
    False when it was already gone (a 404). Any other failure raises."""
    path = f"repos/{repo}/issues/{number}/labels/{urllib.parse.quote(name)}"
    try:
        api("DELETE", path)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise
    return True


def comments(api, repo, number):
    """Every comment of issue or PR `number`, each as {id, user (login), body},
    under `paged`'s cap."""
    return [
        {"id": c.get("id"), "user": (c.get("user") or {}).get("login"), "body": c.get("body")}
        for c in paged(api, f"repos/{repo}/issues/{number}/comments")
    ]


def graphql(api, query, variables):
    """The `data` member of a GraphQL answer. An answer carrying `errors` raises
    RuntimeError rather than be read as partial data."""
    answer = api("POST", "graphql", {"query": query, "variables": variables})
    if answer.get("errors"):
        raise RuntimeError(f"GraphQL lookup failed: {answer['errors']}")
    return answer["data"]


def _positive_int(value):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _lowered(headers):
    """Header names lower-cased: urllib's are case-insensitive, a plain dict is not."""
    return {str(k).lower(): v for k, v in (headers or {}).items()}


def retry_delay(status, headers, body, attempt, now):
    """Seconds to wait before attempt `attempt + 1`, or None to raise now.

    A 403 shares its status with a plain permission failure, so only the
    headers or the body tell a throttle apart: `Retry-After` is what the
    secondary limiter sends, an exhausted primary limit sends
    `x-ratelimit-remaining: 0` with a reset epoch, and absent both a 403 whose
    body does not name the secondary limit is real and must surface."""
    lower = _lowered(headers)
    retry_after = _positive_int(lower.get("retry-after"))
    if status in RETRIED_5XX:
        return min(retry_after, MAX_DELAY) if retry_after else min(2**attempt, MAX_DELAY)
    if status not in THROTTLED:
        return None
    if retry_after is not None:
        return min(retry_after, MAX_DELAY)
    if lower.get("x-ratelimit-remaining") == "0":
        reset = _positive_int(lower.get("x-ratelimit-reset"))
        return min(max(reset - int(now), 1), MAX_DELAY) if reset else MAX_DELAY
    if status == 429 or SECONDARY_LIMIT_HINT in (body or "").lower():
        return min(2**attempt, MAX_DELAY)
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Surface a redirect as an HTTPError instead of following it, so the
    credentials never reach the redirect's host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def urllib_transport(request, timeout, redirects):
    """The real transport: urllib, following redirects or not."""
    if redirects:
        return urllib.request.urlopen(request, timeout=timeout)
    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def _error_body(exc):
    try:
        return exc.read().decode("utf-8", "replace")
    except Exception:
        return ""


class Client:
    """`client(method, path, body=None)` -> the decoded JSON answer.

    An empty body is `{}` for a write or a 204, and a ValueError for any other GET:
    an empty document read as `{}` would pass for a PR with no hold on it.

    `transport(request, timeout, redirects)` is injectable for tests; `sleep` and
    `now` are injectable so the retry policy can be asserted without waiting."""

    def __init__(self, token, *, user_agent, sleep=time.sleep, now=time.time, transport=None):
        self._token = token
        self._user_agent = user_agent
        self._sleep = sleep
        self._now = now
        self._transport = transport or urllib_transport

    def _request(self, method, path, body=None):
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": self._user_agent,
        }
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return urllib.request.Request(
            f"{API_ROOT}/{path}", data=data, headers=headers, method=method
        )

    def _attempts(self, method, path, once, idempotent=True):
        """`once()` until it returns, under the retry policy. A non-retried
        HTTPError is re-raised unchanged; any other exception is never retried."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return once()
            except urllib.error.HTTPError as exc:
                body = _error_body(exc)
                delay = retry_delay(exc.code, exc.headers, body, attempt, self._now())
                if not idempotent and exc.code in RETRIED_5XX:
                    delay = None
                if delay is None or attempt == MAX_ATTEMPTS:
                    if exc.code != 404:
                        line = " ".join(body.split())[:500]
                        print(f"{method} {path} -> HTTP {exc.code}: {line}", file=sys.stderr)
                    raise
                word = "throttled" if exc.code in THROTTLED else "retrying"
                print(f"{word} {method} {path} in {delay}s (HTTP {exc.code})", file=sys.stderr)
                self._sleep(delay)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def __call__(self, method, path, body=None, *, idempotent=True):
        request = self._request(method, path, body)

        def once():
            with self._transport(request, TIMEOUT, True) as resp:
                raw = resp.read()
                status = getattr(resp, "status", None)
            if raw:
                return json.loads(raw)
            if method != "GET" or status == 204:
                return {}
            raise ValueError(f"empty response body for GET {path}")

        return self._attempts(method, path, once, idempotent)

    def download(self, path, read):
        """`read(response)` of an artifact-style download.

        The authenticated GET runs under the retry policy without following
        redirects; an https redirect is that leg's result, never retried or
        raised. The redirect is then followed once, without credentials (the
        blob URL is pre-signed), and any error there propagates unretried. An
        exception from `read` is never retried."""
        request = self._request("GET", path)

        def once():
            try:
                with self._transport(request, DOWNLOAD_TIMEOUT, False) as resp:
                    return False, read(resp)
            except urllib.error.HTTPError as exc:
                location = _lowered(exc.headers).get("location") or ""
                if exc.code in REDIRECTS and location.startswith("https://"):
                    exc.close()  # it holds the authenticated leg's connection
                    return True, location
                raise

        redirected, result = self._attempts("GET", path, once)
        if not redirected:
            return result
        follow = urllib.request.Request(result, headers={"User-Agent": self._user_agent})
        with self._transport(follow, DOWNLOAD_TIMEOUT, True) as resp:
            return read(resp)
