"""PyTorch wheel options derived from the installed NVIDIA driver."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Any

from lorahub.api.system_stats import _find_nvidia_smi


@dataclass(frozen=True, slots=True)
class TorchWheelOption:
    cuda: str
    torch_version: str
    torchvision_version: str
    label: str
    min_driver: str
    notes: str


# Keep this list small and biased toward versions LoraHub already installs.
# PyTorch publishes these wheels under https://download.pytorch.org/whl/{cuda}.
TORCH_WHEEL_OPTIONS: tuple[TorchWheelOption, ...] = (
    TorchWheelOption(
        cuda="cu128",
        torch_version="2.7.1",
        torchvision_version="0.22.1",
        label="PyTorch 2.7.1 / CUDA 12.8",
        min_driver="570.00",
        notes="需要 570+ NVIDIA 驱动。",
    ),
    TorchWheelOption(
        cuda="cu126",
        torch_version="2.6.0",
        torchvision_version="0.21.0",
        label="PyTorch 2.6.0 / CUDA 12.6",
        min_driver="560.00",
        notes="需要 560+ NVIDIA 驱动。",
    ),
    TorchWheelOption(
        cuda="cu124",
        torch_version="2.6.0",
        torchvision_version="0.21.0",
        label="PyTorch 2.6.0 / CUDA 12.4",
        min_driver="550.54",
        notes="需要 550.54+ NVIDIA 驱动。",
    ),
    TorchWheelOption(
        cuda="cu121",
        torch_version="2.5.1",
        torchvision_version="0.20.1",
        label="PyTorch 2.5.1 / CUDA 12.1",
        min_driver="530.30",
        notes="需要 530.30+ NVIDIA 驱动。",
    ),
    TorchWheelOption(
        cuda="cu118",
        torch_version="2.5.1",
        torchvision_version="0.20.1",
        label="PyTorch 2.5.1 / CUDA 11.8",
        min_driver="520.61",
        notes="需要 520.61+ NVIDIA 驱动。",
    ),
)

_CUDA_MIN_DRIVER = {
    option.cuda: option.min_driver for option in TORCH_WHEEL_OPTIONS
} | {
    "cu130": "580.00",
}
_DETECT_DRIVER = object()


def _version_tuple(version: str | None) -> tuple[int, ...] | None:
    if not version:
        return None
    parts = re.findall(r"\d+", version)
    if not parts:
        return None
    return tuple(int(p) for p in parts[:4])


def _version_gte(actual: str | None, minimum: str) -> bool | None:
    actual_tuple = _version_tuple(actual)
    minimum_tuple = _version_tuple(minimum)
    if actual_tuple is None or minimum_tuple is None:
        return None
    width = max(len(actual_tuple), len(minimum_tuple))
    return actual_tuple + (0,) * (width - len(actual_tuple)) >= minimum_tuple + (
        0,
    ) * (width - len(minimum_tuple))


def detect_nvidia_driver() -> str | None:
    """Return the first NVIDIA driver version reported by nvidia-smi."""

    smi = _find_nvidia_smi()
    if not smi:
        return None
    try:
        result = subprocess.run(
            [
                smi,
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        value = line.strip()
        if value:
            return value
    return None


def supports_cuda(driver_version: str | None, cuda: str) -> bool | None:
    """Return whether *driver_version* can load the requested CUDA wheel.

    ``None`` means the host driver is unknown or the CUDA suffix is not in
    LoraHub's compatibility table, so callers should avoid hard-blocking.
    """

    minimum = _CUDA_MIN_DRIVER.get(cuda)
    if not minimum:
        return None
    return _version_gte(driver_version, minimum)


def get_torch_options(driver_version: str | None | object = _DETECT_DRIVER) -> dict[str, Any]:
    driver = (
        detect_nvidia_driver()
        if driver_version is _DETECT_DRIVER
        else driver_version
    )
    if not isinstance(driver, str):
        driver = None
    rows: list[dict[str, Any]] = []
    recommended_index: int | None = None
    max_cuda: str | None = None

    for idx, option in enumerate(TORCH_WHEEL_OPTIONS):
        supported = supports_cuda(driver, option.cuda)
        compatible = True if supported is None else supported
        if recommended_index is None and compatible:
            recommended_index = idx
            max_cuda = option.cuda
        if driver is None:
            reason = "未检测到 NVIDIA 驱动，按默认兼容项展示；安装前请确认 nvidia-smi 可用。"
        elif compatible:
            reason = f"当前驱动 {driver} 满足最低要求 {option.min_driver}。"
        else:
            reason = f"当前驱动 {driver} 低于最低要求 {option.min_driver}。"
        rows.append(
            {
                "cuda": option.cuda,
                "torch_version": option.torch_version,
                "torchvision_version": option.torchvision_version,
                "label": option.label,
                "min_driver": option.min_driver,
                "compatible": compatible,
                "recommended": False,
                "reason": reason,
                "notes": option.notes,
            }
        )

    if recommended_index is not None:
        rows[recommended_index]["recommended"] = True

    return {
        "driver_version": driver,
        "max_cuda": max_cuda,
        "options": rows,
    }


def validate_torch_selection(cuda: str) -> str | None:
    """Return an error string when the selected CUDA wheel is driver-incompatible."""

    driver = detect_nvidia_driver()
    supported = supports_cuda(driver, cuda)
    if supported is False:
        minimum = _CUDA_MIN_DRIVER.get(cuda)
        return (
            f"当前 NVIDIA 驱动 {driver} 低于 {cuda} wheel 的最低要求 {minimum}，"
            "请选择更低 CUDA 版本的 PyTorch 或升级驱动。"
        )
    return None
