"""Cross-field consistency rules for the kohya backend.

Same shape as ``lorahub.core.backends.anima_lora.policies`` — emit
``ValidationIssue`` records for combinations of kohya-relevant fields
that train.py either asserts against at startup or silently degrades.
The rules cover the most-frequent confusions reported in the kohya
issue tracker and on the LoraHub Discord.

We intentionally only enforce rules that hold across **every** arch
kohya supports (sdxl / sd15 / sd3 / flux / lumina / chroma / hunyuan
image / qwen-image). Arch-specific assertions belong in the compiler
(``compile_config`` raises ``CompilationError``); this module is for
the static-shape conflicts a form / API can flag *before* the user
hits "save".
"""

from __future__ import annotations

import importlib.util
from typing import Iterable

from lorahub.core.backends.base import Severity, ValidationIssue
from lorahub.core.config.schema import TrainingConfig


def check_cross_field_conflicts(cfg: TrainingConfig) -> list[ValidationIssue]:
    """Return every cross-field issue ``cfg`` triggers under the
    kohya backend.

    Errors are emitted in priority order — schedule / network rules
    first, then optimizer, then dataset. UIs that only render the
    first few still see the highest-impact items.

    Skipped silently when ``cfg.backend.type`` isn't kohya — the
    rule set assumes kohya semantics and would generate noise on
    anima / diffusion-pipe configs that share the same OptimizerConfig
    section but resolve fields differently downstream.
    """
    if cfg.backend is not None and cfg.backend.type and cfg.backend.type != "kohya":
        return []

    issues: list[ValidationIssue] = []

    issues.extend(_network_conflicts(cfg))
    issues.extend(_optimizer_conflicts(cfg))
    issues.extend(_schedule_conflicts(cfg))
    issues.extend(_dataset_conflicts(cfg))
    issues.extend(_precision_conflicts(cfg))
    issues.extend(_attention_conflicts(cfg))
    return issues


# ---------------------------------------------------------------------- #
# Network / LoRA rank consistency
# ---------------------------------------------------------------------- #


def _network_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    network = cfg.network
    if network is None:
        return

    rank = int(network.rank or 0)
    alpha = float(network.alpha or 0)
    if rank < 1:
        yield ValidationIssue(
            Severity.error,
            "network.rank",
            f"network.rank={rank} 必须 >= 1。LoRA 不能学一个零维空间。",
        )
        return

    ratio = alpha / max(rank, 1)
    if ratio >= 4.0 or ratio <= 0.25:
        yield ValidationIssue(
            Severity.warning,
            "network.alpha",
            f"network.alpha={alpha} 与 rank={rank} 比值 {ratio:.2f} 偏离推荐区间 "
            f"[0.25, 4.0]。effective LR ∝ alpha/rank,过大 LoRA 漂移、过小不学;"
            "推荐 alpha == rank,或 0.5×~2× 之间。",
        )

    if rank < 4:
        yield ValidationIssue(
            Severity.warning,
            "network.rank",
            f"network.rank={rank} 过低,LoRA 学不到内容。SDXL character LoRA "
            "通常 16-32,style LoRA 8-16。",
        )


# ---------------------------------------------------------------------- #
# Optimizer / LR
# ---------------------------------------------------------------------- #


def _optimizer_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    opt = cfg.optimizer
    if opt is None:
        return

    optimizer_type = (opt.type or "").strip()
    is_8bit = optimizer_type.lower().endswith("8bit")
    is_fused = "fused" in optimizer_type.lower()

    if is_8bit and not _has_bitsandbytes():
        yield ValidationIssue(
            Severity.warning,
            "optimizer.type",
            f"optimizer.type='{optimizer_type}' 需要 bitsandbytes,但当前活跃 venv "
            "里检测不到这个包。在 sd-scripts 子环境里也许有,但若运行时缺失训练会以 "
            "'CUDA Setup failed' 退出。`pip install bitsandbytes` 或在 sd-scripts "
            "venv 里装。",
        )

    if is_fused and cfg.precision not in ("bf16", "fp16"):
        yield ValidationIssue(
            Severity.warning,
            "optimizer.type",
            f"optimizer.type='{optimizer_type}' (fused) 通常配 bf16/fp16 才有意义;"
            f"当前 precision={cfg.precision}。fp32 + fused 不会更快也不省显存。",
        )

    # te lr 是死字段的检测对 kohya 走 cfg.backend.extra_args 里的
    # network_train_unet_only。schema 顶层没这个字段, 所以这里读 extra_args。
    extras = cfg.backend.extra_args if cfg.backend is not None else None
    train_unet_only = bool(extras.get("network_train_unet_only")) if extras else False
    te_lr = opt.lr.text_encoder
    unet_lr = opt.lr.unet
    user_set_te = (
        te_lr is not None
        and te_lr != 5.0e-5  # schema default
        and te_lr != unet_lr
    )
    if train_unet_only and user_set_te:
        yield ValidationIssue(
            Severity.info,
            "optimizer.lr.textEncoder",
            f"backend.extraArgs.network_train_unet_only=true 时 textEncoder LR ({te_lr}) "
            "不生效 (冻结)。建议把 textEncoder LR 与 unet LR 设成一样,或留 schema 默认 5e-5。",
        )


def _has_bitsandbytes() -> bool:
    try:
        return importlib.util.find_spec("bitsandbytes") is not None
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------- #
# Schedule / save cadence
# ---------------------------------------------------------------------- #


def _schedule_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    sch = cfg.schedule
    if sch is None:
        return

    bs = max(int(sch.batch_size or 1), 1)
    accum = max(int(sch.grad_accum or 1), 1)
    eff_batch = bs * accum
    if eff_batch > 64:
        yield ValidationIssue(
            Severity.warning,
            "schedule.gradAccum",
            f"effective batch (batchSize × gradAccum = {bs}×{accum} = {eff_batch}) "
            "偏大,LoRA 微调常见的 effective batch 在 4~16,过大会让梯度信号过度平均。",
        )

    save_every = getattr(cfg.output, "save_every_n_epochs", None) if cfg.output else None
    max_epochs = getattr(sch, "epochs", None)
    if save_every and max_epochs and save_every > max_epochs:
        yield ValidationIssue(
            Severity.warning,
            "output.saveEveryNEpochs",
            f"output.save_every_n_epochs={save_every} 大于 schedule.epochs={max_epochs},"
            "整次训练只会在最后落一次盘,没有中间 checkpoint。建议设到 epochs 的 1/4~1/2。",
        )


# ---------------------------------------------------------------------- #
# Dataset / bucket / caption
# ---------------------------------------------------------------------- #


def _dataset_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    ds = cfg.dataset
    if ds is None:
        return

    res = ds.resolution
    if isinstance(res, (list, tuple)) and len(res) == 2:
        long_edge = max(int(res[0]), int(res[1]))
    else:
        long_edge = int(res or 0)

    bucket = ds.bucket
    if bucket is not None and bucket.enabled:
        bmin = int(bucket.min_size or 0)
        bmax = int(bucket.max_size or 0)
        if bmin and bmax and bmin > bmax:
            yield ValidationIssue(
                Severity.error,
                "dataset.bucket.min",
                f"dataset.bucket.min={bmin} 大于 max={bmax},区间为空 — kohya 启动会抛 "
                "AssertionError 'bucket reso list is empty'。",
            )
        if bmax and long_edge and bmax > long_edge * 2:
            yield ValidationIssue(
                Severity.warning,
                "dataset.bucket.max",
                f"dataset.bucket.max={bmax} 比 resolution 长边 {long_edge} 大 2 倍以上,"
                "极端长宽比图片会进巨大的 bucket,显存占用呈平方增长,容易 OOM。",
            )

    cap = ds.caption
    if cap is not None:
        keep = cap.keep_tokens
        shuffle = cap.shuffle
        if keep is not None and keep > 0 and not shuffle:
            yield ValidationIssue(
                Severity.info,
                "dataset.caption.keepTokens",
                f"dataset.caption.keepTokens={keep} 但 shuffle=false,这个值不起作用 "
                "(没有 shuffle 就没有要锚定的对象)。",
            )
        drop_rate = cap.drop_rate
        if drop_rate is not None and drop_rate >= 0.5:
            yield ValidationIssue(
                Severity.warning,
                "dataset.caption.dropRate",
                f"dataset.caption.dropRate={drop_rate} 过高,每步至少一半样本完全丢弃 "
                "caption,模型会迅速忘掉 trigger word。常用值在 0.05-0.15。",
            )


# ---------------------------------------------------------------------- #
# Precision
# ---------------------------------------------------------------------- #


def _precision_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    base = cfg.base_model
    if base is None:
        return

    # fp8 text_encoder dtype 与全局 fp32 precision 同时存在是 footgun ——
    # 用户大概率没意识到 fp8 会让 te 输出走 fp8 量化。
    te_dtype = getattr(base, "t5xxl_dtype", None) or getattr(base, "text_encoder_dtype", None)
    if te_dtype == "fp8" and cfg.precision == "fp32":
        yield ValidationIssue(
            Severity.warning,
            "baseModel.t5xxlDtype",
            "text encoder dtype 设为 fp8 但全局 precision 是 fp32,组合不常见 — "
            "fp8 te 通常配 bf16/fp16 训练。检查这是不是有意为之。",
        )


# ---------------------------------------------------------------------- #
# Attention impl
# ---------------------------------------------------------------------- #


def _attention_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    """attn flags — sdpa / xformers / flash 三选一冲突最常见。"""
    extras = cfg.backend.extra_args if cfg.backend is not None else None
    if not extras:
        return

    flags_on = []
    if extras.get("xformers"):
        flags_on.append("xformers")
    if extras.get("sdpa"):
        flags_on.append("sdpa")
    if extras.get("flash_attn") or extras.get("mem_eff_attn"):
        flags_on.append("flash/mem_eff")

    if len(flags_on) > 1:
        yield ValidationIssue(
            Severity.warning,
            "backend.extraArgs",
            f"同时开了多种 attention 实现 ({', '.join(flags_on)}),"
            "kohya 只取最先识别到的那个,其余被忽略。建议只留一个。",
        )


__all__ = ["check_cross_field_conflicts"]
