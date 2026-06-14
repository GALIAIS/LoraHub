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
        types.SimpleNamespace(rfilename="README.md", size=100),
        types.SimpleNamespace(rfilename="preview.png", size=256),
        types.SimpleNamespace(rfilename="config.json", size=12),
        types.SimpleNamespace(rfilename="model.safetensors", size=4096),
        types.SimpleNamespace(rfilename="tokenizer/vocab.txt", size=64),
    ]

    class FakeApi:
        def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
            self.endpoint = endpoint
            self.token = token

        def model_info(
            self,
            repo_id: str,
            revision: str,
            files_metadata: bool,
            token: str | None = None,
        ):
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
        endpoint: str | None = None,
        token: str | None = None,
    ) -> str:
        worker_names.add(threading.current_thread().name)
        assert filename not in {"README.md", "preview.png"}
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

    expected = [s for s in siblings if s.rfilename not in {"README.md", "preview.png"}]
    assert result.files == len(expected)
    assert result.total_bytes == sum(s.size for s in expected)
    # We expect: list-files event, file-count event, one per finished file, plus the final event.
    per_file_events = [e for e in events if e.files_done and e.files_done >= 1]
    assert len(per_file_events) >= len(expected)
    assert events[-1].percent == 100
    assert events[-1].files_done == len(expected)
    # Multi-threaded: at least one file ran on a non-main thread when threads > 1.
    assert any(name != threading.current_thread().name for name in worker_names)


def test_huggingface_download_honours_explicit_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    siblings = [
        types.SimpleNamespace(rfilename="model-a.safetensors", size=10),
        types.SimpleNamespace(rfilename="model-b.safetensors", size=20),
        types.SimpleNamespace(rfilename="README.md", size=30),
    ]

    class FakeApi:
        def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
            pass

        def model_info(self, *_args: Any, **_kwargs: Any):
            return types.SimpleNamespace(siblings=siblings)

    downloaded: list[str] = []

    def fake_hf_hub_download(**kw: Any) -> str:
        filename = kw["filename"]
        downloaded.append(filename)
        out = Path(kw["local_dir"]) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return str(out)

    fake_hub = types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    result = downloader.download(
        DownloadRequest(
            source="huggingface",
            repo_id="owner/name",
            target_dir=tmp_path / "hf",
            paths=("model-b.safetensors",),
        )
    )

    assert downloaded == ["model-b.safetensors"]
    assert result.files == 1
    assert result.total_bytes == 20


def test_huggingface_download_fails_when_any_selected_file_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    siblings = [
        types.SimpleNamespace(rfilename="ok.safetensors", size=10),
        types.SimpleNamespace(rfilename="broken.safetensors", size=20),
    ]

    class FakeApi:
        def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
            pass

        def model_info(self, *_args: Any, **_kwargs: Any):
            return types.SimpleNamespace(siblings=siblings)

    def fake_hf_hub_download(**kw: Any) -> str:
        filename = kw["filename"]
        if filename == "broken.safetensors":
            raise RuntimeError("network reset")
        out = Path(kw["local_dir"]) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 10)
        return str(out)

    fake_hub = types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    events: list[downloader.DownloadProgress] = []
    with pytest.raises(RuntimeError, match="broken.safetensors"):
        downloader.download(
            DownloadRequest(
                source="huggingface",
                repo_id="owner/name",
                target_dir=tmp_path / "hf",
                threads=2,
            ),
            events.append,
        )

    assert any("failed" in event.message for event in events)


def test_huggingface_download_uses_env_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://hf-mirror.com/")
    monkeypatch.delenv("HUGGINGFACE_HUB_ENDPOINT", raising=False)
    endpoints: list[str | None] = []
    siblings = [types.SimpleNamespace(rfilename="model.safetensors", size=7)]

    class FakeApi:
        def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
            endpoints.append(endpoint)

        def model_info(self, *_args: Any, **_kwargs: Any):
            return types.SimpleNamespace(siblings=siblings)

    def fake_hf_hub_download(**kw: Any) -> str:
        endpoints.append(kw.get("endpoint"))
        out = Path(kw["local_dir"]) / kw["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 7)
        return str(out)

    fake_hub = types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    events: list[downloader.DownloadProgress] = []
    downloader.download(
        DownloadRequest(
            source="huggingface",
            repo_id="owner/name",
            target_dir=tmp_path / "hf",
        ),
        events.append,
    )

    assert endpoints == ["https://hf-mirror.com", "https://hf-mirror.com"]
    assert "https://hf-mirror.com" in events[0].message


def test_huggingface_explicit_endpoint_wins_over_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_ENDPOINT", "https://env-mirror.example")
    endpoints: list[str | None] = []
    siblings = [types.SimpleNamespace(rfilename="model.safetensors", size=1)]

    class FakeApi:
        def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
            endpoints.append(endpoint)

        def model_info(self, *_args: Any, **_kwargs: Any):
            return types.SimpleNamespace(siblings=siblings)

    def fake_hf_hub_download(**kw: Any) -> str:
        endpoints.append(kw.get("endpoint"))
        out = Path(kw["local_dir"]) / kw["filename"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        return str(out)

    fake_hub = types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_hf_hub_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    downloader.download(
        DownloadRequest(
            source="huggingface",
            repo_id="owner/name",
            target_dir=tmp_path / "hf",
            huggingface_endpoint="https://settings-mirror.example/",
        )
    )

    assert endpoints == [
        "https://settings-mirror.example",
        "https://settings-mirror.example",
    ]


def test_list_remote_files_marks_default_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    files = [
        ("README.md", 10),
        ("preview.jpg", 20),
        ("model.safetensors", 30),
        ("tokenizer/vocab.txt", 40),
    ]
    monkeypatch.setattr(downloader, "_hf_list_files", lambda *_args, **_kw: files)

    listed = downloader.list_remote_files(
        DownloadRequest(source="huggingface", repo_id="owner/name")
    )

    selected = {f.path for f in listed if f.selected}
    assert selected == {"model.safetensors", "tokenizer/vocab.txt"}
    assert next(f for f in listed if f.path == "README.md").reason == "ignored by default"


def test_download_refuses_when_selection_is_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    siblings = [
        types.SimpleNamespace(rfilename="README.md", size=10),
        types.SimpleNamespace(rfilename="preview.jpg", size=20),
    ]

    class FakeApi:
        def __init__(self, endpoint: str | None = None, token: str | None = None) -> None:
            pass

        def model_info(self, *_args: Any, **_kwargs: Any):
            return types.SimpleNamespace(siblings=siblings)

    fake_hub = types.SimpleNamespace(
        HfApi=FakeApi,
        hf_hub_download=lambda **_kw: pytest.fail("must not download"),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    with pytest.raises(ValueError, match="no files selected"):
        downloader.download(
            DownloadRequest(
                source="huggingface",
                repo_id="owner/name",
                target_dir=tmp_path / "hf",
            )
        )


def test_modelscope_download_uses_parallel_workers_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        {"Path": "README.md", "Size": 100},
        {"Path": "preview.png", "Size": 200},
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
        assert file_path not in {"README.md", "preview.png"}
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


def test_modelscope_download_fails_when_any_selected_file_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        {"Path": "ok.bin", "Size": 3},
        {"Path": "broken.bin", "Size": 4},
    ]

    def list_files(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return files

    def download_file(
        _repo_id: str,
        _revision: str,
        file_path: str,
        target: Path,
        _token: str | None,
    ) -> int:
        if file_path == "broken.bin":
            raise RuntimeError("connection closed")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"ok")
        return 2

    monkeypatch.setattr(downloader, "_ms_list_files", list_files)
    monkeypatch.setattr(downloader, "_ms_download_file", download_file)

    events: list[downloader.DownloadProgress] = []
    with pytest.raises(RuntimeError, match="broken.bin"):
        downloader.download(
            DownloadRequest(
                source="modelscope",
                repo_id="owner/name",
                target_dir=tmp_path / "ms",
                threads=2,
            ),
            events.append,
        )

    assert any("failed" in event.message for event in events)


def test_modelscope_file_download_does_not_leave_partial_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.safetensors"

    class BrokenResponse:
        def __init__(self) -> None:
            self._reads = 0

        def __enter__(self) -> "BrokenResponse":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def read(self, _size: int) -> bytes:
            self._reads += 1
            if self._reads == 1:
                return b"partial"
            raise RuntimeError("connection reset")

    monkeypatch.setattr(downloader, "urlopen", lambda *_args, **_kwargs: BrokenResponse())

    with pytest.raises(RuntimeError, match="connection reset"):
        downloader._ms_download_file(
            "owner/name",
            "master",
            "model.safetensors",
            target,
            None,
        )

    assert not target.exists()
    assert not target.with_name("model.safetensors.part").exists()
