"""The shared GitHub client of `.github/scripts` (#2068): one request shape, one
retry policy, one capped paginator and one artifact download for every caller.

The policy literals below (5 attempts, 2/4/8/16 s, the 60 s cap, timeouts of 30 s
and 60 s) are the ones the #2068 plan locked; the headers are GitHub's REST
contract (docs.github.com/en/rest/using-the-rest-api/getting-started-with-the-rest-api).
"""

import re
import urllib.error
from pathlib import Path

import pytest

from ._github_fake import FakeResponse, FakeTransport, http_error, load_client

gh = load_client()

NOW = 1_000_000
URL = "https://api.github.com/repos/o/r/pulls/7"
ZIP = "repos/o/r/actions/artifacts/1/zip"
ZIP_URL = f"https://api.github.com/{ZIP}"
BLOB = "https://blob.example/z"
SECONDARY = "You have exceeded a secondary rate limit"


def _client(script):
    sleeps = []
    transport = FakeTransport(script)
    client = gh.Client(
        "t0ken",
        user_agent="test-agent",
        sleep=sleeps.append,
        now=lambda: NOW,
        transport=transport,
    )
    return client, transport, sleeps


# --- retry_delay: the one table every request goes through -------------------


@pytest.mark.parametrize(
    ("status", "headers", "body", "attempt", "expected"),
    [
        (403, {"Retry-After": "120"}, "", 1, 60),
        (403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW + 7)}, "", 1, 7),
        (403, {"x-ratelimit-remaining": "0"}, "", 1, 60),
        (429, {}, "", 3, 8),
        (403, {}, SECONDARY, 1, 2),
        (403, {}, "Resource not accessible by integration", 1, None),
        (500, {}, "", 1, None),
        (302, {"Location": BLOB}, "", 1, None),
        (503, {}, "", 2, 4),
        (503, {"Retry-After": "9"}, "", 1, 9),
        (403, {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(NOW - 10)}, "", 1, 1),
    ],
    ids=[
        "retry-after-capped",
        "exhausted-waits-for-reset",
        "exhausted-without-reset",
        "429-backs-off",
        "403-secondary-backs-off",
        "plain-403-raises",
        "500-raises",
        "redirect-raises",
        "503-backs-off",
        "503-honours-retry-after",
        "exhausted-reset-in-the-past-floors-at-1s",
    ],
)
def test_retry_delay(status, headers, body, attempt, expected):
    assert gh.retry_delay(status, headers, body, attempt, NOW) == expected


# --- __call__ ---------------------------------------------------------------


def test_a_request_is_authenticated_versioned_and_bounded_by_a_timeout():
    client, transport, _ = _client([{"number": 7}, {}])
    assert client("GET", "repos/o/r/pulls/7") == {"number": 7}
    client("POST", "repos/o/r/issues/7/labels", {"labels": ["x"]})
    get, post = transport.requests
    assert (get.url, get.method, get.timeout) == (URL, "GET", 30)
    assert get.headers == {
        "authorization": "Bearer t0ken",
        "accept": "application/vnd.github+json",
        "x-github-api-version": "2022-11-28",
        "user-agent": "test-agent",
    }
    assert get.body is None
    assert (post.method, post.body) == ("POST", {"labels": ["x"]})
    assert post.headers["content-type"] == "application/json"


def test_an_empty_body_is_an_empty_object_only_for_a_write_or_a_204():
    """An empty 2xx body on a GET is no document: read as `{}` it is a PR with no
    reviewer and no label, and waiting-human-review went green (#2068 review). A
    write's empty answer, and a 204, are `{}`."""
    client, _, _ = _client([b"", b"", FakeResponse(b"", status=204)])
    assert client("POST", "repos/o/r/issues/7/labels", {"labels": ["x"]}) == {}
    with pytest.raises(ValueError, match=r"^empty response body for GET repos/o/r/pulls/7$"):
        client("GET", "repos/o/r/pulls/7")
    assert client("GET", "repos/o/r/pulls/7/merge") == {}


def test_a_5xx_is_retried_with_doubling_backoff_until_it_succeeds():
    client, transport, sleeps = _client([http_error(503)] * 4 + [{"ok": 1}])
    assert client("GET", "repos/o/r/pulls/7") == {"ok": 1}
    assert sleeps == [2, 4, 8, 16]
    assert len(transport.requests) == 5


def test_a_5xx_that_persists_raises_after_five_attempts():
    client, transport, sleeps = _client([http_error(503)] * 5)
    with pytest.raises(urllib.error.HTTPError) as info:
        client("GET", "repos/o/r/pulls/7")
    assert info.value.code == 503
    assert len(transport.requests) == 5
    assert sleeps == [2, 4, 8, 16]


def test_a_plain_403_raises_at_once_and_logs_its_body(capsys):
    client, transport, sleeps = _client([http_error(403, "Resource not accessible")])
    with pytest.raises(urllib.error.HTTPError) as info:
        client("GET", "repos/o/r/pulls/7")
    assert info.value.code == 403
    assert (len(transport.requests), sleeps) == (1, [])
    assert capsys.readouterr().err == (
        "GET repos/o/r/pulls/7 -> HTTP 403: Resource not accessible\n"
    )


def test_a_secondary_rate_limit_is_waited_out_and_logged(capsys):
    client, _, sleeps = _client([http_error(403, SECONDARY), {"ok": 1}])
    assert client("GET", "repos/o/r/pulls/7") == {"ok": 1}
    assert sleeps == [2]
    assert capsys.readouterr().err == "throttled GET repos/o/r/pulls/7 in 2s (HTTP 403)\n"


def test_a_404_raises_unchanged_and_logs_nothing(capsys):
    """A 404 is an answer some callers expect (a label already gone): no log line."""
    client, transport, sleeps = _client([http_error(404, "Not Found")])
    with pytest.raises(urllib.error.HTTPError) as info:
        client("GET", "repos/o/r/pulls/7")
    assert info.value.code == 404
    assert (len(transport.requests), sleeps) == (1, [])
    assert capsys.readouterr().err == ""


def test_a_final_error_logs_its_body_so_a_rejected_token_shows_why(capsys):
    """#2057: a rejected token or missing permission ended as a bare status."""
    client, _, _ = _client([http_error(401, '{"message":"Bad credentials"}')])
    with pytest.raises(urllib.error.HTTPError):
        client("GET", "repos/o/r/pulls/7")
    assert capsys.readouterr().err == (
        'GET repos/o/r/pulls/7 -> HTTP 401: {"message":"Bad credentials"}\n'
    )


def test_a_logged_error_body_is_collapsed_to_one_line(capsys):
    """A body line starting `::` would be read by the runner as a workflow command."""
    client, _, _ = _client([http_error(422, "x\n::error::y")])
    with pytest.raises(urllib.error.HTTPError):
        client("POST", "repos/o/r/issues/7/comments", {"body": "b"})
    assert capsys.readouterr().err == (
        "POST repos/o/r/issues/7/comments -> HTTP 422: x ::error::y\n"
    )


def test_a_non_idempotent_request_is_not_retried_on_a_5xx():
    """A 5xx after a committed write would write it twice (a second ledger row)."""
    client, transport, sleeps = _client([http_error(502), {"id": 1}])
    with pytest.raises(urllib.error.HTTPError) as info:
        client("POST", "repos/o/r/issues/7/comments", {"body": "b"}, idempotent=False)
    assert info.value.code == 502
    assert (len(transport.requests), sleeps) == (1, [])


def test_a_non_idempotent_request_still_waits_out_a_throttle():
    """A throttled request was never processed, so retrying it writes once."""
    client, transport, sleeps = _client([http_error(429), {"id": 1}])
    assert client("POST", "repos/o/r/issues/7/comments", {"b": 1}, idempotent=False) == {"id": 1}
    assert (len(transport.requests), sleeps) == (2, [2])


# --- remove_label, comments, graphql ---------------------------------------------


def test_remove_label_deletes_the_quoted_label_and_answers_true():
    client, transport, _ = _client([FakeResponse(b"", status=204)])
    assert gh.remove_label(client, "o/r", 7, "status:review") is True
    (req,) = transport.requests
    assert (req.method, req.url) == (
        "DELETE",
        "https://api.github.com/repos/o/r/issues/7/labels/status%3Areview",
    )


def test_remove_label_answers_false_when_the_label_is_already_gone():
    client, _, _ = _client([http_error(404)])
    assert gh.remove_label(client, "o/r", 7, "status:review") is False


def test_remove_label_raises_any_other_failure():
    client, _, _ = _client([http_error(422)])
    with pytest.raises(urllib.error.HTTPError) as info:
        gh.remove_label(client, "o/r", 7, "status:review")
    assert info.value.code == 422


def test_comments_maps_each_comment_to_id_user_and_body():
    page = [{"id": 1, "user": {"login": "bot"}, "body": "x", "node_id": "N"}]
    client, transport, _ = _client([page])
    assert gh.comments(client, "o/r", 7) == [{"id": 1, "user": "bot", "body": "x"}]
    assert transport.requests[0].url == (
        "https://api.github.com/repos/o/r/issues/7/comments?per_page=100&page=1"
    )


def test_comments_raises_when_ten_full_pages_may_go_on():
    full = [{"id": i, "user": {"login": "u"}, "body": ""} for i in range(100)]
    client, transport, _ = _client([full] * 10)
    with pytest.raises(gh.ListingTruncated):
        gh.comments(client, "o/r", 7)
    assert len(transport.requests) == 10


def test_graphql_raises_on_an_errors_answer():
    client, _, _ = _client([{"errors": [{"message": "bad"}], "data": None}])
    with pytest.raises(RuntimeError, match=r"^GraphQL lookup failed: \[\{'message': 'bad'\}\]$"):
        gh.graphql(client, "query { x }", {"n": 1})


def test_graphql_posts_the_query_and_returns_its_data():
    client, transport, _ = _client([{"data": {"repository": {"id": "R"}}}])
    assert gh.graphql(client, "query { x }", {"n": 1}) == {"repository": {"id": "R"}}
    (req,) = transport.requests
    assert (req.method, req.url) == ("POST", "https://api.github.com/graphql")
    assert req.body == {"query": "query { x }", "variables": {"n": 1}}


# --- download ---------------------------------------------------------------


def _redirect(location=BLOB):
    return http_error(302, headers={"Location": location}, url=ZIP_URL)


def test_download_of_a_direct_body_hands_the_response_to_read():
    client, transport, sleeps = _client({ZIP_URL: [FakeResponse(b"PK-zip")]})
    assert client.download(ZIP, lambda resp: resp.read()) == b"PK-zip"
    (leg,) = transport.requests
    assert (leg.timeout, leg.redirects) == (60, False)
    assert leg.headers["authorization"] == "Bearer t0ken"
    assert sleeps == []


def test_download_follows_one_https_redirect_without_credentials():
    client, transport, sleeps = _client({ZIP_URL: [_redirect()], BLOB: [FakeResponse(b"PK")]})
    assert client.download(ZIP, lambda resp: resp.read()) == b"PK"
    leg, follow = transport.requests
    assert (leg.url, leg.timeout, leg.redirects) == (ZIP_URL, 60, False)
    assert (follow.url, follow.timeout, follow.redirects) == (BLOB, 60, True)
    assert follow.headers == {"user-agent": "test-agent"}
    assert sleeps == []


def test_download_closes_the_redirect_response():
    """The redirect's HTTPError holds the socket of the authenticated leg."""
    redirect = _redirect()
    client, _, _ = _client({ZIP_URL: [redirect], BLOB: [FakeResponse(b"PK")]})
    client.download(ZIP, lambda resp: resp.read())
    assert redirect.fp.closed is True


def test_download_retries_the_authenticated_leg_on_a_5xx():
    client, _, sleeps = _client(
        {ZIP_URL: [http_error(503), _redirect()], BLOB: [FakeResponse(b"PK")]}
    )
    assert client.download(ZIP, lambda resp: resp.read()) == b"PK"
    assert sleeps == [2]


def test_download_waits_out_a_secondary_rate_limit_on_the_authenticated_leg():
    client, _, sleeps = _client(
        {ZIP_URL: [http_error(403, SECONDARY), _redirect()], BLOB: [FakeResponse(b"PK")]}
    )
    assert client.download(ZIP, lambda resp: resp.read()) == b"PK"
    assert sleeps == [2]


def test_download_does_not_retry_the_follow_up():
    client, transport, sleeps = _client({ZIP_URL: [_redirect()], BLOB: [http_error(503)]})
    with pytest.raises(urllib.error.HTTPError) as info:
        client.download(ZIP, lambda resp: resp.read())
    assert info.value.code == 503
    assert [r.url for r in transport.requests] == [ZIP_URL, BLOB]
    assert sleeps == []


def test_download_refuses_a_redirect_off_https():
    client, transport, sleeps = _client({ZIP_URL: [_redirect("http://blob.example/z")]})
    with pytest.raises(urllib.error.HTTPError) as info:
        client.download(ZIP, lambda resp: resp.read())
    assert info.value.code == 302
    assert [r.url for r in transport.requests] == [ZIP_URL]
    assert sleeps == []


def test_download_does_not_retry_when_read_rejects_the_body():
    class Rejected(Exception):
        pass

    def read(resp):
        raise Rejected

    client, transport, sleeps = _client({ZIP_URL: [FakeResponse(b"PK")]})
    with pytest.raises(Rejected):
        client.download(ZIP, read)
    assert (len(transport.requests), sleeps) == (1, [])


# --- paged: one capped paginator (#2076, moved from code_gate_verdict) ----------


def _pages(sizes):
    """A list endpoint whose page N holds sizes[N-1] items (then nothing)."""
    calls = []

    def api(method, path, body=None):
        calls.append(path)
        page = int(re.search(r"[?&]page=(\d+)", path).group(1))
        size = sizes[page - 1] if page <= len(sizes) else 0
        return [{"n": (page, i)} for i in range(size)]

    return api, calls


def test_paged_raises_naming_the_listing_when_its_last_allowed_page_is_full():
    """A full last page may go on: handing it on as the whole list is fail-open."""
    api, calls = _pages([100, 100, 100, 7])
    with pytest.raises(RuntimeError, match=r"repos/o/r/things.*truncated") as info:
        gh.paged(api, "repos/o/r/things", max_pages=3)
    assert isinstance(info.value, gh.ListingTruncated)
    assert len(info.value.items) == 300
    assert len(calls) == 3


def test_paged_returns_a_listing_that_ends_inside_its_cap():
    api, calls = _pages([100, 100, 40])
    assert len(gh.paged(api, "repos/o/r/things", max_pages=3)) == 240
    assert calls == [
        "repos/o/r/things?per_page=100&page=1",
        "repos/o/r/things?per_page=100&page=2",
        "repos/o/r/things?per_page=100&page=3",
    ]


def test_paged_ends_at_an_empty_page():
    api, calls = _pages([100, 0])
    assert len(gh.paged(api, "repos/o/r/things", max_pages=3)) == 100
    assert len(calls) == 2


@pytest.mark.parametrize(
    ("page", "key", "got"),
    [
        ({}, None, "dict"),
        ({"artifacts": {}}, "artifacts", "dict"),
        ({"total_count": 0}, "artifacts", "NoneType"),
        ([], "artifacts", "NoneType"),
    ],
    ids=["object-for-a-list", "object-under-the-key", "key-missing", "list-for-an-object"],
)
def test_paged_refuses_a_page_that_is_not_a_list(page, key, got):
    """A page that is not a list, read as an empty one, ends the listing as if it
    were whole: a bot release or a dismissal past it goes unseen (#2068 review)."""
    calls = []

    def api(method, path, body=None):
        calls.append(path)
        return page

    with pytest.raises(ValueError, match=rf"^listing repos/o/r/things page 1: .* got {got}$"):
        gh.paged(api, "repos/o/r/things", key=key)
    assert calls == ["repos/o/r/things?per_page=100&page=1"]


def test_the_client_takes_its_token_as_an_argument_and_reads_no_environment():
    """#2079 injects its own token: a client that read the environment would
    silently use the job's token instead."""
    source = (Path(gh.__file__)).read_text(encoding="utf-8")
    assert re.findall(r"os\.environ|getenv", source) == []
