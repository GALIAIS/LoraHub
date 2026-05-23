"""Outgoing pipeline for the error report registry.

The local SQLite store is the source of truth; this package adds an
*optional* fan-out to a remote sink so users with a shared
``LoraHubReport`` GitLab project (or a custom webhook) can collect
failures off the box.

Pieces:
    * ``redaction`` — strip secrets / user paths / emails / IPs from
      every outgoing payload before it leaves the machine.
    * ``fingerprint`` — stable hash so the GitLab sink can de-dupe
      repeated occurrences into a single issue with appended comments.
    * ``sinks`` — concrete senders. ``GitLabIssueSink`` opens / appends
      to GitLab issues; ``WebhookSink`` POSTs JSON to any URL.
    * ``dispatcher`` — async queue + retry policy bridging the local
      store and a configured sink.

No code path here is reachable unless the user explicitly configures a
sink in Settings → 错误上报. Default ``channel="off"`` means *nothing*
ever leaves the box.
"""

from __future__ import annotations

from .dispatcher import UpstreamDispatcher
from .fingerprint import compute_fingerprint
from .redaction import redact_report
from .sinks import (
    GiteaIssueSink,
    GitLabIssueSink,
    SendResult,
    SinkConfig,
    UpstreamSink,
    WebhookSink,
    build_sink_from_settings,
)

__all__ = [
    "GiteaIssueSink",
    "GitLabIssueSink",
    "SendResult",
    "SinkConfig",
    "UpstreamDispatcher",
    "UpstreamSink",
    "WebhookSink",
    "build_sink_from_settings",
    "compute_fingerprint",
    "redact_report",
]
