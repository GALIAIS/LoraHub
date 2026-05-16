from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from typing import Any

import pytest

from lorahub.core.models import downloader
from lorahub.core.models.downloader import DownloadRequest


def test_huggingface_download_emits_per_file_progress_and_uses_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    siblings = [
        types.SimpleNamespace(rfilename="config.json", size=12),
        types.SimpleNamespace(rfilename="model.safetensors", size=4096),
        types.SimpleNamespace(rfilename="tokenizer/vocab.txt", size=64),
    ]

    class FakeApi:
        def model_info(self, repo_id: str, revision: str, files_metadata: bool):
            assert repo_id == "owner/name"
            assert revision == "main"
            assert files_metadata is True
            return types.SimpleNamespace(siblings=siblings)

    worker_names: set[str] = set()

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        revision: str,
        local_dir: str,
    ) -> str:
        worker_names.add(threading.current_thread().name)
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * next(s.size for s in siblings if s.rfilename == filename))
        return str(out)

    fake_hub = types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    events: list[downloader.DownloadProgress] = []
    result = downloader.download(
        DownloadRequest(
            source="huggingface",
            repo_id="owner/name",
            revision="main",
            target_dir=tmp_path / "hf",
            threads=3,
        ),
        events.append,
    )

    assert result.files == len(siblings)
    assert result.total_bytes == sum(s.size for s in siblings)
    # We expect: list-files event, file-count event, one per finished file, plus the final event.
    per_file_events = [e for e in events if e.files_done and e.files_done >= 1]
    assert len(per_file_events) >= len(siblings)
    assert events[-1].percent == 100
    assert events[-1].files_done == len(siblings)
    # Multi-threaded: at least one file ran on a non-main thread when threads > 1.
    assert any(name != threading.current_thread().name for name in worker_names)


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
