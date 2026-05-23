"""Tests for the upstream fan-out layer.

Three layers under test:
    * Redaction — secrets / paths / emails / IPs are stripped from
      every textual surface in an ErrorReport.
    * Fingerprint — same crash → same hash; line numbers and
      occurrence-specific noise don't bust the cache.
    * Sinks (GitLabIssueSink + WebhookSink) — both their happy paths
      and their failure-mode classification (retryable vs not).

We never make real HTTP requests; both sinks reach the network through
``lorahub.api.error_upstream.sinks._http`` so we monkeypatch that
helper to return canned responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from lorahub.api.error_reports import ErrorReport
from lorahub.api.error_upstream import (
    GitLabIssueSink,
    WebhookSink,
    compute_fingerprint,
    redact_report,
)
from lorahub.api.error_upstream import sinks as sinks_module
from lorahub.api.error_upstream.dispatcher import UpstreamDispatcher
from lorahub.api.error_upstream.redaction import _redact_text
from lorahub.api.error_upstream.sinks import SendResult


# --------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------- #


def test_redact_strips_authorization_headers() -> None:
    text = "Authorization: Bearer abcdef1234567890ZZZZZZZZ"
    out = _redact_text(text)
    assert "Bearer" not in out
    assert "abcdef1234567890" not in out
    assert "REDACTED" in out


def test_redact_strips_inline_secrets() -> None:
    text = "config has api_key=sk-1234567890abcdef and password=verysecret"
    out = _redact_text(text)
    assert "sk-1234567890abcdef" not in out
    assert "verysecret" not in out


def test_redact_collapses_user_home_to_tilde() -> None:
    win = "C:\\Users\\alice\\Documents\\runs\\foo.log"
    posix = "/home/alice/runs/foo.log"
    mac = "/Users/alice/runs/foo.log"
    assert _redact_text(win).startswith("~\\Documents")
    assert _redact_text(posix).startswith("~/runs")
    assert _redact_text(mac).startswith("~/runs")


def test_redact_collapses_drive_root() -> None:
    text = "loaded F:\\D\\LoraHub\\models\\foo.safetensors"
    out = _redact_text(text)
    assert "F:\\D\\" not in out
    assert "<drive>:\\" in out


def test_redact_email_and_ip() -> None:
    out = _redact_text("alice@example.com talks to 203.0.113.42 over IPv4")
    assert "<email>" in out
    assert "<ip>" in out
    # Loopback is preserved so triage isn't blinded to "this came from
    # the same machine" failures.
    assert "127.0.0.1" in _redact_text("connect to 127.0.0.1:18765")


def test_redact_report_walks_every_textual_surface() -> None:
    rep = ErrorReport.create(
        severity="error",
        source="backend.job",
        category="x",
        title="failure at /home/alice/runs/run-1",
        message="Authorization: Bearer xxxxxxxxxxxxxxxxxxxx",
        stack="C:\\Users\\alice\\code\\app.py line 12 in oops",
        context={
            "headers": {"Authorization": "Bearer secrettoken1234567890"},
            "host": "203.0.113.7",
            "email": "alice@example.com",
        },
        request_path="/api/jobs",
    )
    redacted = redact_report(rep)
    assert "alice" not in redacted.title
    assert "Bearer" not in redacted.message
    assert "alice" not in (redacted.stack or "")
    headers = redacted.context["headers"]
    assert isinstance(headers, dict)
    # Key-name based scrubbing: even though the value happens to contain
    # the word ``Bearer``, the dict lookup *on a key named Authorization*
    # must reduce the whole leaf to ***REDACTED***.
    assert "secrettoken" not in headers["Authorization"]
    assert "<ip>" in redacted.context["host"]
    assert "<email>" in redacted.context["email"]


# --------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------- #


def _make_report(**kw: Any) -> ErrorReport:
    base = dict(
        severity="error",
        source="backend.job",
        category="x",
        title="t",
        message="m",
    )
    base.update(kw)
    return ErrorReport.create(**base)


def test_fingerprint_is_stable_across_occurrences() -> None:
    a = _make_report(stack="File \"x.py\", line 10, in foo")
    b = _make_report(stack="File \"x.py\", line 11, in foo")
    # Different line numbers in the same function: same fingerprint.
    assert compute_fingerprint(a) == compute_fingerprint(b)


def test_fingerprint_changes_with_function_name() -> None:
    a = _make_report(stack="File \"x.py\", line 10, in foo")
    b = _make_report(stack="File \"x.py\", line 10, in bar")
    assert compute_fingerprint(a) != compute_fingerprint(b)


def test_fingerprint_ignores_uuid_in_message() -> None:
    a = _make_report(
        message="job 7c3a8f1e-1234-4567-89ab-1234567890ab failed at step 1234",
    )
    b = _make_report(
        message="job 99fea2bc-aaaa-bbbb-cccc-ddddeeeeffff failed at step 9999",
    )
    assert compute_fingerprint(a) == compute_fingerprint(b)


# --------------------------------------------------------------------- #
# GitLabIssueSink
# --------------------------------------------------------------------- #


@pytest.fixture
def gitlab_sink() -> GitLabIssueSink:
    return GitLabIssueSink(
        base_url="https://git.example.com",
        repo_path="space/proj",
        token="tok",
    )


def test_gitlab_open_new_issue_when_no_match(
    monkeypatch: pytest.MonkeyPatch, gitlab_sink: GitLabIssueSink,
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_http(url: str, **kw: Any):
        method = kw.get("method", "GET")
        calls.append((method, url))
        if method == "GET" and "/issues?" in url:
            # No prior issue with this fingerprint.
            return 200, []
        if method == "POST" and url.endswith("/issues"):
            return 201, {"iid": 42, "web_url": "https://git.example.com/space/proj/issues/42"}
        raise AssertionError(f"unexpected call: {method} {url}")

    monkeypatch.setattr(sinks_module, "_http", fake_http)
    rep = _make_report()
    res = gitlab_sink.send(rep)
    assert res.ok
    assert res.upstream_id == "42"
    assert res.url.endswith("/issues/42")
    methods = [m for m, _ in calls]
    assert methods == ["GET", "POST"]


def test_gitlab_appends_comment_on_existing_issue(
    monkeypatch: pytest.MonkeyPatch, gitlab_sink: GitLabIssueSink,
) -> None:
    def fake_http(url: str, **kw: Any):
        method = kw.get("method", "GET")
        if method == "GET" and "/issues?" in url:
            return 200, [{
                "iid": 17,
                "user_notes_count": 3,
                "web_url": "https://git.example.com/space/proj/issues/17",
            }]
        if method == "POST" and url.endswith("/notes"):
            return 201, {"id": 99}
        raise AssertionError(f"unexpected call: {method} {url}")

    monkeypatch.setattr(sinks_module, "_http", fake_http)
    res = gitlab_sink.send(_make_report())
    assert res.ok
    assert res.upstream_id == "17"
    assert "issues/17" in res.url


def test_gitlab_5xx_is_retryable(
    monkeypatch: pytest.MonkeyPatch, gitlab_sink: GitLabIssueSink,
) -> None:
    def fake_http(url: str, **kw: Any):
        method = kw.get("method", "GET")
        if method == "GET":
            return 200, []
        return 502, "bad gateway"

    monkeypatch.setattr(sinks_module, "_http", fake_http)
    res = gitlab_sink.send(_make_report())
    assert res.ok is False
    assert res.retryable is True


def test_gitlab_4xx_is_not_retryable(
    monkeypatch: pytest.MonkeyPatch, gitlab_sink: GitLabIssueSink,
) -> None:
    def fake_http(url: str, **kw: Any):
        method = kw.get("method", "GET")
        if method == "GET":
            return 200, []
        return 401, "invalid token"

    monkeypatch.setattr(sinks_module, "_http", fake_http)
    res = gitlab_sink.send(_make_report())
    assert res.ok is False
    assert res.retryable is False


# --------------------------------------------------------------------- #
# WebhookSink
# --------------------------------------------------------------------- #


def test_webhook_post_redacts_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_http(url: str, **kw: Any):
        captured["url"] = url
        captured["body"] = kw.get("body")
        captured["headers"] = kw.get("headers", {})
        return 200, {"ok": True}

    monkeypatch.setattr(sinks_module, "_http", fake_http)
    sink = WebhookSink(url="https://hooks.example.com/x", auth_header="Bearer abc")
    rep = _make_report(message="api_key=sk-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    res = sink.send(rep)
    assert res.ok
    body = captured["body"].decode("utf-8")
    assert "sk-aaa" not in body
    assert "REDACTED" in body
    assert captured["headers"]["Authorization"] == "Bearer abc"


def test_webhook_network_error_is_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sinks_module, "_http",
        lambda *a, **kw: (-1, "URLError(timeout)"),
    )
    sink = WebhookSink(url="https://hooks.example.com/x")
    res = sink.send(_make_report())
    assert res.ok is False
    assert res.retryable is True


# --------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------- #


class _StubSink:
    channel = "webhook"

    def __init__(self, results: list[SendResult]):
        self.results = results
        self.calls = 0

    def send(self, report: ErrorReport) -> SendResult:  # type: ignore[override]
        self.calls += 1
        return self.results[min(self.calls - 1, len(self.results) - 1)]

    def health_check(self) -> SendResult:  # type: ignore[override]
        return SendResult(ok=True)


def test_dispatcher_send_now_marks_sent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from lorahub.api.error_reports import ErrorReportStore

    store = ErrorReportStore(tmp_path / "errors.sqlite")
    sink = _StubSink([SendResult(ok=True, upstream_id="55", url="https://x/55")])
    dispatcher = UpstreamDispatcher(store=store, sink_factory=lambda: sink)
    rep = _make_report()
    store.insert(rep)
    res = dispatcher.send_now(rep)
    assert res.ok
    persisted = store.get(rep.id)
    assert persisted is not None
    assert persisted.upstream_status == "sent"
    assert persisted.upstream_url == "https://x/55"
    assert persisted.sent_at is not None


def test_dispatcher_send_now_records_failed_when_channel_off(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from lorahub.api.error_reports import ErrorReportStore

    store = ErrorReportStore(tmp_path / "errors.sqlite")
    dispatcher = UpstreamDispatcher(store=store, sink_factory=lambda: None)
    rep = _make_report()
    store.insert(rep)
    res = dispatcher.send_now(rep)
    assert res.ok is False
    persisted = store.get(rep.id)
    assert persisted is not None
    assert persisted.upstream_status == "failed"
