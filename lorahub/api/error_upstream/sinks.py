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


SinkChannel = Literal["off", "gitlab", "gitea", "webhook"]


@dataclass
class SinkConfig:
    """User-supplied configuration for the upstream sink.

    Only fields the matching channel needs are required; the others
    are tolerated as empty strings so the same dataclass can be
    serialised straight from / to ``Settings``.

    GitLab and Gitea share the ``gitlab_*`` field set because both
    consume the same three values (base URL + ``owner/repo`` path +
    personal access token). The channel discriminator picks the
    matching sink class at construction time.
    """

    channel: SinkChannel = "off"
    # GitLab / Gitea fields
    gitlab_base_url: str = ""        # e.g. ``https://git.galiais.com``
    gitlab_repo: str = ""            # e.g. ``Shiro/LoraHubReport``
    gitlab_token: str = ""           # PAT with ``api`` (gitlab) or
                                     # ``write:issue`` (gitea) scope
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
# Gitea Issues
# ---------------------------------------------------------------------- #


@dataclass
class GiteaIssueSink:
    """File reports as Gitea issues with the same fingerprint contract.

    Gitea's issue API mirrors GitHub's, not GitLab's:

    * Base path is ``/api/v1`` (vs GitLab's ``/api/v4``).
    * Auth header is ``Authorization: token <pat>`` (vs GitLab's
      ``PRIVATE-TOKEN``).
    * Repo is ``{owner}/{repo}`` and **must not** be URL-encoded —
      Gitea's router parses the slash directly.
    * Issues key off ``number`` rather than GitLab's ``iid``.
    * Labels are addressed by **integer id**, not by name. We
      lazily ``GET /labels`` to discover existing ones, ``POST
      /labels`` for the missing ones, and cache the resulting
      name → id map per process so subsequent reports skip the
      round-trip.
    * Search by label uses ``GET /issues/search?type=issues
      &state=open&labels=fp:<hash>`` — Gitea accepts label *names*
      in the query string here even though attaching them needs
      ids.
    """

    base_url: str
    repo_path: str          # ``owner/repo``
    token: str
    timeout_s: float = 12.0
    max_comments_per_issue: int = 50

    channel: SinkChannel = field(default="gitea", init=False)
    # Lazy cache of label name → numeric id. Populated by
    # ``_ensure_labels`` on the first send; tests can pre-seed it via
    # the public attribute to skip the discovery round-trips.
    _label_cache: dict[str, int] = field(default_factory=dict, init=False)

    # ------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------ #

    def send(self, report: ErrorReport) -> SendResult:
        if not self.base_url or not self.repo_path or not self.token:
            return SendResult(ok=False, error="gitea sink not configured", retryable=False)
        redacted = redact_report(report)
        fp = compute_fingerprint(redacted)
        existing = self._find_issue_by_fingerprint(fp)
        if existing is not None:
            number = int(existing["number"])
            web_url = str(existing.get("html_url") or "")
            comments_count = int(existing.get("comments") or 0)
            if comments_count >= self.max_comments_per_issue:
                cont_fp = f"{fp}-cont{(comments_count // self.max_comments_per_issue)}"
                return self._open_new_issue(redacted, cont_fp)
            return self._append_comment(number, redacted, web_url)
        return self._open_new_issue(redacted, fp)

    def health_check(self) -> SendResult:
        if not self.base_url or not self.repo_path or not self.token:
            return SendResult(ok=False, error="gitea sink not configured", retryable=False)
        # Probe the issues endpoint, not the repo metadata endpoint —
        # the repo metadata route requires ``read:repository`` scope
        # while users only need to grant ``read:issue`` + ``write:issue``
        # for this sink to function. Asking for repo scope on top of
        # those would be over-broad and would fail closed for tokens
        # that follow the principle of least privilege.
        url = f"{self._repo_url()}/issues?limit=1&state=open&type=issues"
        status, body = _http(
            url, headers=self._headers(), timeout_s=self.timeout_s,
        )
        if status != 200:
            retryable = status >= 500 or status == -1
            return SendResult(
                ok=False,
                error=f"Gitea health probe failed ({status}): {body!r}"[:300],
                retryable=retryable,
            )
        # Bonus check: ``git.galiais.com`` was observed running on a
        # database whose column charset isn't utf8mb4 — any non-ASCII
        # text gets silently rewritten to ``?`` server-side. The probe
        # uploads a *throwaway* issue with a CJK marker, reads it back,
        # and warns when the marker survived as question marks. We
        # delete the probe issue regardless of outcome so it doesn't
        # pollute the registry. The check is best-effort: if any of
        # these calls fails (token cannot create / read / delete) the
        # connectivity result is still ``ok`` — only the encoding hint
        # is dropped.
        encoding_hint = self._probe_unicode_round_trip()
        repo_html_url = (
            f"{self.base_url.rstrip('/')}/{self.repo_path.strip().strip('/')}"
        )
        if encoding_hint:
            return SendResult(
                ok=True,
                url=repo_html_url,
                error=encoding_hint,  # surface as a warning in the UI
            )
        return SendResult(ok=True, url=repo_html_url)

    def _probe_unicode_round_trip(self) -> str:
        """Round-trip a CJK marker through the issues API.

        Returns an explanation string when the round-trip is lossy —
        i.e. the database column can't hold non-ASCII text and the
        marker comes back as ``?`` characters. Empty string means
        either the round-trip survived intact or the probe couldn't
        complete (network blip / quota / lack of delete permission)
        and we'd rather stay quiet than false-alarm.
        """
        marker = "你好-Unicode-Probe"
        body = json.dumps(
            {
                "title": "lorahub-encoding-probe (auto-delete)",
                "body": marker,
            }
        ).encode("utf-8")
        post_status, post_body = _http(
            f"{self._repo_url()}/issues",
            method="POST",
            headers=self._headers(),
            body=body,
            timeout_s=self.timeout_s,
        )
        if post_status not in (200, 201) or not isinstance(post_body, dict):
            return ""
        number = post_body.get("number")
        stored_body = str(post_body.get("body") or "")
        # Cleanup first (best-effort) so an exception in the assertion
        # doesn't leave the probe issue dangling.
        if isinstance(number, int):
            close_payload = json.dumps({"state": "closed"}).encode("utf-8")
            _http(
                f"{self._repo_url()}/issues/{number}",
                method="PATCH",
                headers=self._headers(),
                body=close_payload,
                timeout_s=self.timeout_s,
            )
            _http(
                f"{self._repo_url()}/issues/{number}",
                method="DELETE",
                headers=self._headers(),
                timeout_s=self.timeout_s,
            )
        if marker not in stored_body:
            return (
                "连通正常,但服务端把非 ASCII 字符替换为 '?'(可能数据库未配置 "
                "utf8mb4 / 列编码不支持中文)。issue 标题与正文中的中文与符号会丢失。"
                "请联系 Gitea 管理员升级数据库编码。"
            )
        return ""

    # ------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------ #

    def _repo_url(self) -> str:
        # Gitea expects the slash to remain literal — quoting the
        # whole ``owner/repo`` returns 404. We do strip leading /
        # trailing whitespace defensively because users paste from
        # the address bar.
        return f"{self.base_url.rstrip('/')}/api/v1/repos/{self.repo_path.strip().strip('/')}"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"token {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "lorahub-error-reporter",
        }

    def _fingerprint_label(self, fp: str) -> str:
        return f"fp:{fp}"

    def _find_issue_by_fingerprint(self, fp: str) -> dict[str, Any] | None:
        # Gitea's per-repo issue search accepts ``labels=<name>`` as a
        # CSV; ``state=open`` keeps a manually-closed dupe from
        # silently re-opening as a new issue.
        params = urllib.parse.urlencode(
            {
                "type": "issues",
                "state": "open",
                "labels": self._fingerprint_label(fp),
                "limit": "1",
            },
        )
        url = f"{self._repo_url()}/issues?{params}"
        status, body = _http(url, headers=self._headers(), timeout_s=self.timeout_s)
        if status == 200 and isinstance(body, list) and body:
            return body[0]
        return None

    def _ensure_labels(self, names: list[str]) -> list[int]:
        """Resolve label names to ids, creating any missing entries.

        Cached at the instance level so a chatty run only pays the
        discovery cost once. Failure to create a label is non-fatal —
        we drop that label rather than fail the whole send.
        """
        result: list[int] = []
        unresolved: list[str] = []
        for name in names:
            cached = self._label_cache.get(name)
            if cached is not None:
                result.append(cached)
            else:
                unresolved.append(name)
        if not unresolved:
            return result

        # Pull the full label list once and refresh the cache.
        list_url = f"{self._repo_url()}/labels?limit=200"
        status, body = _http(list_url, headers=self._headers(), timeout_s=self.timeout_s)
        if status == 200 and isinstance(body, list):
            for entry in body:
                if isinstance(entry, dict) and "name" in entry and "id" in entry:
                    self._label_cache[str(entry["name"])] = int(entry["id"])
        # Anything still missing — create. ``color`` is required by
        # Gitea even if we don't care about it; pick a stable shade
        # per severity / source rather than random so a manual review
        # of the label list isn't visually noisy.
        for name in unresolved:
            if name in self._label_cache:
                result.append(self._label_cache[name])
                continue
            create_body = json.dumps({
                "name": name,
                "color": _label_colour_for(name),
            }).encode("utf-8")
            create_status, create_body_resp = _http(
                f"{self._repo_url()}/labels",
                method="POST",
                headers=self._headers(),
                body=create_body,
                timeout_s=self.timeout_s,
            )
            if create_status in (200, 201) and isinstance(create_body_resp, dict):
                self._label_cache[name] = int(create_body_resp["id"])
                result.append(self._label_cache[name])
            else:
                log.warning(
                    "could not create gitea label %r (%s): %r",
                    name, create_status, create_body_resp,
                )
        return result

    def _open_new_issue(self, report: ErrorReport, fp: str) -> SendResult:
        title = f"[{report.severity}] {report.title[:200]}"
        label_names = sorted(
            {
                self._fingerprint_label(fp),
                f"severity:{report.severity}",
                f"source:{report.source}",
                f"category:{report.category}",
            }
        )
        label_ids = self._ensure_labels(label_names)
        body = json.dumps(
            {
                "title": title,
                "body": _render_markdown(report, fingerprint=fp),
                "labels": label_ids,
            }
        ).encode("utf-8")
        url = f"{self._repo_url()}/issues"
        status, payload = _http(
            url, method="POST", headers=self._headers(),
            body=body, timeout_s=self.timeout_s,
        )
        # Defensive validation. Gitea has been observed to return 200
        # with a stub body when the token lacks the right scopes —
        # in that case ``number`` is still echoed but no issue lands
        # in the repo. We treat a missing ``html_url`` as a hard
        # failure so the dispatcher records ``failed`` instead of a
        # silent ``sent``.
        if status in (200, 201) and isinstance(payload, dict):
            number = payload.get("number")
            html_url = payload.get("html_url")
            if number is None or not html_url:
                return SendResult(
                    ok=False,
                    error=(
                        f"Gitea accepted POST but did not echo a usable "
                        f"issue body — token may lack ``read:issue`` "
                        f"scope. Response: {payload!r}"
                    )[:500],
                    retryable=False,
                )
            return SendResult(
                ok=True,
                upstream_id=str(number),
                url=str(html_url),
            )
        retryable = status >= 500 or status == -1 or status == 429
        return SendResult(
            ok=False,
            error=f"Gitea create issue failed ({status}): {payload!r}"[:500],
            retryable=retryable,
        )

    def _append_comment(
        self, number: int, report: ErrorReport, issue_url: str,
    ) -> SendResult:
        body = json.dumps(
            {"body": _render_markdown(report, fingerprint=None, head_level=4)}
        ).encode("utf-8")
        url = f"{self._repo_url()}/issues/{number}/comments"
        status, payload = _http(
            url, method="POST", headers=self._headers(),
            body=body, timeout_s=self.timeout_s,
        )
        if status in (200, 201) and isinstance(payload, dict):
            return SendResult(
                ok=True,
                upstream_id=str(number),
                url=issue_url,
            )
        retryable = status >= 500 or status == -1 or status == 429
        return SendResult(
            ok=False,
            error=f"Gitea append comment failed ({status}): {payload!r}"[:500],
            retryable=retryable,
        )


_LABEL_COLOURS = {
    "severity:fatal": "#7f1d1d",
    "severity:error": "#b91c1c",
    "severity:warn": "#b45309",
    "severity:info": "#0e7490",
    "source:backend.exception": "#374151",
    "source:backend.job": "#1f2937",
    "source:frontend.render": "#5b21b6",
    "source:frontend.runtime": "#6d28d9",
    "source:frontend.api": "#7c3aed",
    "source:user.report": "#0f766e",
}


def _label_colour_for(label: str) -> str:
    """Stable colour token so the Gitea issues list stays scannable.

    Falls back to a neutral grey for the dynamic ``fp:<hash>`` and
    ``category:<x>`` labels — those would generate noise if every new
    one picked a fresh hue.
    """
    return _LABEL_COLOURS.get(label, "#64748b")


def _render_markdown(
    report: ErrorReport, fingerprint: str | None = None, *, head_level: int = 3,
) -> str:
    """Issue / comment body shared by GitLab and Gitea sinks.

    Lifted out of ``GitLabIssueSink._render_body`` so a future
    GitHub / Gitea-flavoured fork doesn't have to duplicate the
    marker layout. Same fields, same headings.
    """
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
    if cfg.channel == "gitea":
        return GiteaIssueSink(
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
    "GiteaIssueSink",
    "SendResult",
    "SinkChannel",
    "SinkConfig",
    "UpstreamSink",
    "WebhookSink",
    "build_sink_from_settings",
]
