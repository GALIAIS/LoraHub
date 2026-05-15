from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from lorahub.core.models import downloader
from lorahub.core.models.downloader import DownloadRequest


def test_huggingface_download_uses_progress_and_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def snapshot_download(**kwargs: Any) -> None:
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.safetensors").write_bytes(b"weights")

    fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    events: list[downloader.DownloadProgress] = []
    result = downloader.download(
        DownloadRequest(
            source="huggingface",
            repo_id="owner/name",
            revision="main",
            target_dir=tmp_path / "hf",
            threads=7,
        ),
        events.append,
    )

    assert result.files == 1
    assert result.total_bytes == len(b"weights")
    assert calls[0]["max_workers"] == 7
    assert events[0].percent == 5
    assert events[-1].percent == 100


def test_modelscope_download_uses_parallel_workers_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        {"Path": "a.bin", "Size": 3},
        {"Path": "nested/b.bin", "Size": 4},
    ]
    worker_names: set[str] = set()

    def list_files(repo_id: str, revision: str, token: str | None) -> list[dict[str, Any]]:
        assert repo_id == "owner/name"
        assert revision == "master"
        assert token == "secret"
        return files

    def download_file(
        repo_id: str,
        revision: str,
        file_path: str,
        target: Path,
        token: str | None,
    ) -> int:
        worker_names.add(threading.current_thread().name)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = file_path.encode("utf-8")
        target.write_bytes(payload)
        return len(payload)

    monkeypatch.setattr(downloader, "_ms_list_files", list_files)
    monkeypatch.setattr(downloader, "_ms_download_file", download_file)

    events: list[downloader.DownloadProgress] = []
    result = downloader.download(
        DownloadRequest(
            source="modelscope",
            repo_id="owner/name",
            revision="master",
            target_dir=tmp_path / "ms",
            modelscope_token="secret",
            threads=2,
        ),
        events.append,
    )

    assert result.files == 2
    assert result.total_bytes == len("a.bin") + len("nested/b.bin")
    assert (tmp_path / "ms" / "a.bin").is_file()
    assert (tmp_path / "ms" / "nested" / "b.bin").is_file()
    assert any(event.files_done == 2 and event.percent == 100 for event in events)
    assert worker_names
