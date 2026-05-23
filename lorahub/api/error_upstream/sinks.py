"""Concrete error-report sinks.

Two flavours so users can pick the one that fits their infra:

* :class:`GitLabIssueSink` — open / append to issues on a self-hosted
  or saas GitLab project. Carries fingerprint-based de-dupe so a
  recurring crash collapses into one issue with a comment per
  occurrence instead of N separate issues.
* :class:`WebhookSink` — flat ``POST <url>`` with the redacted JSON
  body. The simplest possible drop — Slack incoming webhooks, custom
  collectors, n8n flows.

Both implement :class:`UpstreamSink`. Pick the right one through
:func:`build_sink_from_settings`, which reads the user's
``Settings.error_upstream`` block.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from lorahub.api.error_reports import ErrorReport
from .fingerprint import compute_fingerprint
from .redaction import redact_dict, redact_report

log = logging.getLogger(__name__)


SinkChannel = Literal["off", "gitlab", "webhook"]


@dataclass
class SinkConfig:
    """User-supplied configuration for the upstream sink.

    Only fields the matching channel needs are required; the others
    are tolerated as empty strings so the same dataclass can be
    serialised straight from / to ``Settings``.
    """

    channel: SinkChannel = "off"
    # GitLab fields
    gitlab_base_url: str = ""        # e.g. ``https://git.galiais.com``
    gitlab_repo: str = ""            # e.g. ``Shiro/LoraHubReport``
    gitlab_token: str = ""           # PAT with ``api`` scope
    # Webhook fields
    webhook_url: str = ""
    webhook_auth_header: str = ""    # raw header value, e.g. ``Bearer abc``
    # Cross-cutting
    auto_send_severity: Literal["off", "error", "all"] = "error"
    timeout_s: float = 12.0


@dataclass
class SendResult:
    """Outcome of one ``send`` call.

    ``ok`` distinguishes a successful upload from a transient error;
    callers use ``retryable`` to decide whether to schedule a backoff
    (5xx, network timeouts) or to drop and surface the failure to the
    user immediately (4xx — config error). ``url`` and ``upstream_id``
    let the registry record where the report was filed so the UI can
    deep-link back later.
    """

    ok: bool
    upstream_id: str = ""
    url: str = ""
    error: str = ""
    retryable: bool = False


class UpstreamSink(Protocol):
    channel: SinkChannel

    def send(self, report: ErrorReport) -> SendResult: ...

    def health_check(self) -> SendResult:
        """Probe the sink end-to-end: token validity + reachability."""


# ---------------------------------------------------------------------- #
# HTTP helpers
# ---------------------------------------------------------------------- #


def _http(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: bytes | None = None,
    timeout_s: float = 12.0,
) -> tuple[int, dict[str, Any] | str]:
    """Tiny urllib wrapper used by both sinks. Returns (status, decoded body).

    Decoded body is parsed as JSON when content-type permits, else
    raw text. Network failures surface as (-1, error string) so the
    sinks have a uniform branch shape.
    """
    req = urllib.request.Request(  # noqa: S310 — user-configured URL
        url, method=method, headers=headers or {}, data=body,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            raw = resp.read()
            ctype = resp.headers.get("content-type", "")
            if "json" in ctype:
                try:
                    return resp.status, json.loads(raw.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return resp.status, raw.decode("utf-8", errors="replace")
            return resp.status, raw.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        # Surface the body so 4xx debugging is possible without curl.
        try:
            err_body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            err_body = ""
        return exc.code, err_body or exc.reason
    except (urllib.error.URLError, OSError) as exc:
        return -1, repr(exc)


# ---------------------------------------------------------------------- #
# GitLab Issues
# ---------------------------------------------------------------------- #


@dataclass
class GitLabIssueSink:
    """File reports as GitLab issues with fingerprint-based de-dupe.

    The first occurrence of a fingerprint opens a new issue; every
    subsequent occurrence within the registry that matches the same
    fingerprint adds a comment to the same issue, capped at
    ``max_comments_per_issue`` so a runaway error doesn't generate an
    issue with 10 000 comments. After the cap we open a fresh issue
    with the same fingerprint suffixed by ``-cont``.
    """

    base_url: str
    repo_path: str          # ``namespace/project``
    token: str
    timeout_s: float = 12.0
    max_comments_per_issue: int = 50

    channel: SinkChannel = field(default="gitlab", init=False)

    # ------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------ #

    def send(self, report: ErrorReport) -> SendResult:
        if not self.base_url or not self.repo_path or not self.token:
            return SendResult(ok=False, error="gitlab sink not configured", retryable=False)
        redacted = redact_report(report)
        fp = compute_fingerprint(redacted)

        # 1. Look for an existing issue tagged with this fingerprint.
        existing = self._find_issue_by_fingerprint(fp)
        if existing is not None:
            iid = int(existing["iid"])
            web_url = str(existing.get("web_url") or "")
            comments_count = int(existing.get("user_notes_count") or 0)
            if comments_count >= self.max_comments_per_issue:
                # Close the loop: open a continuation issue. Stamp the
                # fp with ``-cont<n>`` so future occurrences keep
                # converging on this latest issue rather than the
                # frozen original.
                cont_fp = f"{fp}-cont{(comments_count // self.max_comments_per_issue)}"
                return self._open_new_issue(redacted, cont_fp)
            # Comment on the existing issue.
            return self._append_comment(iid, redacted, web_url)

        # 2. No prior issue — open a new one.
        return self._open_new_issue(redacted, fp)

    def health_check(self) -> SendResult:
        if not self.base_url or not self.repo_path or not self.token:
            return SendResult(ok=False, error="gitlab sink not configured", retryable=False)
        url = f"{self._project_url()}"
        status, body = _http(
            url,
            headers=self._headers(),
            timeout_s=self.timeout_s,
        )
        if status == 200 and isinstance(body, dict) and "id" in body:
            return SendResult(
                ok=True,
                url=str(body.get("web_url") or url),
                upstream_id=str(body.get("id") or ""),
            )
        retryable = status >= 500 or status == -1
        return SendResult(
            ok=False,
            error=f"GitLab health probe failed ({status}): {body!r}"[:300],
            retryable=retryable,
        )

    # ------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------ #

    def _project_url(self) -> str:
        encoded = urllib.parse.quote(self.repo_path, safe="")
        return f"{self.base_url.rstrip('/')}/api/v4/projects/{encoded}"

    def _headers(self) -> dict[str, str]:
        return {
            "PRIVATE-TOKEN": self.token,
            "Content-Type": "application/json",
            "User-Agent": "lorahub-error-reporter",
        }

    def _fingerprint_label(self, fp: str) -> str:
        return f"fp:{fp}"

    def _find_issue_by_fingerprint(self, fp: str) -> dict[str, Any] | None:
        # GitLab's labels are URL-encoded comma-separated; we ask for
        # state=all so a closed-as-dupe issue still gets re-used.
        label = self._fingerprint_label(fp)
        params = urllib.parse.urlencode(
            {"labels": label, "state": "opened", "per_page": "1"},
        )
        url = f"{self._project_url()}/issues?{params}"
        status, body = _http(url, headers=self._headers(), timeout_s=self.timeout_s)
        if status == 200 and isinstance(body, list) and body:
            return body[0]
        return None

    def _open_new_issue(self, report: ErrorReport, fp: str) -> SendResult:
        title = f"[{report.severity}] {report.title[:200]}"
        labels = ",".join(
            sorted(
                {
                    self._fingerprint_label(fp),
                    f"severity:{report.severity}",
                    f"source:{report.source}",
                    f"category:{report.category}",
                }
            )
        )
        description = self._render_body(report, fp)
        body = json.dumps(
            {"title": title, "description": description, "labels": labels},
        ).encode("utf-8")
        url = f"{self._project_url()}/issues"
        status, payload = _http(
            url, method="POST", headers=self._headers(),
            body=body, timeout_s=self.timeout_s,
        )
        if status in (200, 201) and isinstance(payload, dict):
            return SendResult(
                ok=True,
                upstream_id=str(payload.get("iid") or ""),
                url=str(payload.get("web_url") or ""),
            )
        retryable = status >= 500 or status == -1 or status == 429
        return SendResult(
            ok=False,
            error=f"GitLab create issue failed ({status}): {payload!r}"[:500],
            retryable=retryable,
        )

    def _append_comment(
        self, iid: int, report: ErrorReport, issue_url: str,
    ) -> SendResult:
        body_text = self._render_body(report, fingerprint=None, head_level=4)
        body = json.dumps({"body": body_text}).encode("utf-8")
        url = f"{self._project_url()}/issues/{iid}/notes"
        status, payload = _http(
            url, method="POST", headers=self._headers(),
            body=body, timeout_s=self.timeout_s,
        )
        if status in (200, 201) and isinstance(payload, dict):
            return SendResult(
                ok=True,
                upstream_id=str(iid),
                url=issue_url,
            )
        retryable = status >= 500 or status == -1 or status == 429
        return SendResult(
            ok=False,
            error=f"GitLab append comment failed ({status}): {payload!r}"[:500],
            retryable=retryable,
        )

    def _render_body(
        self, report: ErrorReport, fingerprint: str | None = None, *, head_level: int = 3,
    ) -> str:
        """Markdown body shared by issue creation + comment appends."""
        h = "#" * head_level
        out: list[str] = []
        if fingerprint is not None:
            out.append(f"{h} LoraHub error report")
            out.append("")
            out.append(f"- **fingerprint**: `{fingerprint}`")
        else:
            out.append(f"{h} 重新出现")
            out.append("")
        out.append(f"- **time**: {report.timestamp.isoformat()}")
        out.append(f"- **severity**: {report.severity}")
        out.append(f"- **source**: {report.source}")
        out.append(f"- **category**: {report.category}")
        out.append(f"- **version**: {report.version}")
        out.append(f"- **platform**: {report.platform}")
        if report.job_id:
            out.append(f"- **job_id**: {report.job_id}")
        if report.request_path:
            out.append(f"- **request_path**: {report.request_path}")
        if report.request_id:
            out.append(f"- **request_id**: {report.request_id}")
        out.append(f"- **report_id**: {report.id}")
        out.append("")
        out.append("**Message**")
        out.append("```")
        out.append(report.message)
        out.append("```")
        if report.stack:
            out.append("")
            out.append("**Stack**")
            out.append("```")
            out.append(report.stack[:8000])
            out.append("```")
        if report.context:
            out.append("")
            out.append("**Context**")
            out.append("```json")
            try:
                out.append(json.dumps(report.context, ensure_ascii=False, indent=2)[:8000])
            except (TypeError, ValueError):
                out.append(str(report.context)[:8000])
            out.append("```")
        return "\n".join(out)


# ---------------------------------------------------------------------- #
# Webhook
# ---------------------------------------------------------------------- #


@dataclass
class WebhookSink:
    """POST the redacted report as JSON to a user-configured URL."""

    url: str
    auth_header: str = ""
    timeout_s: float = 12.0
    channel: SinkChannel = field(default="webhook", init=False)

    def send(self, report: ErrorReport) -> SendResult:
        if not self.url:
            return SendResult(ok=False, error="webhook sink not configured", retryable=False)
        redacted = redact_report(report)
        payload = redact_dict(redacted.to_dict())
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lorahub-error-reporter",
        }
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        status, response = _http(
            self.url,
            method="POST",
            headers=headers,
            body=body,
            timeout_s=self.timeout_s,
        )
        if 200 <= status < 300:
            return SendResult(ok=True, upstream_id="", url=self.url)
        retryable = status >= 500 or status == -1 or status == 429
        return SendResult(
            ok=False,
            error=f"webhook POST failed ({status}): {response!r}"[:500],
            retryable=retryable,
        )

    def health_check(self) -> SendResult:
        if not self.url:
            return SendResult(ok=False, error="webhook sink not configured", retryable=False)
        # We can't safely probe an arbitrary webhook with GET — many
        # collectors return 405 / route only POSTs. So health_check
        # does the cheapest possible POST: a single ``ping`` event.
        body = json.dumps(
            {"event": "ping", "source": "lorahub.health_check"}
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "lorahub-error-reporter",
        }
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        status, response = _http(
            self.url,
            method="POST",
            headers=headers,
            body=body,
            timeout_s=self.timeout_s,
        )
        if 200 <= status < 300:
            return SendResult(ok=True, url=self.url)
        retryable = status >= 500 or status == -1
        return SendResult(
            ok=False,
            error=f"webhook ping failed ({status}): {response!r}"[:300],
            retryable=retryable,
        )


# ---------------------------------------------------------------------- #
# Settings → sink wiring
# ---------------------------------------------------------------------- #


def build_sink_from_settings(cfg: SinkConfig) -> UpstreamSink | None:
    """Return a configured sink, or None when the channel is off."""
    if cfg.channel == "off":
        return None
    if cfg.channel == "gitlab":
        return GitLabIssueSink(
            base_url=cfg.gitlab_base_url,
            repo_path=cfg.gitlab_repo,
            token=cfg.gitlab_token,
            timeout_s=cfg.timeout_s,
        )
    if cfg.channel == "webhook":
        return WebhookSink(
            url=cfg.webhook_url,
            auth_header=cfg.webhook_auth_header,
            timeout_s=cfg.timeout_s,
        )
    return None


__all__ = [
    "GitLabIssueSink",
    "SendResult",
    "SinkChannel",
    "SinkConfig",
    "UpstreamSink",
    "WebhookSink",
    "build_sink_from_settings",
]
