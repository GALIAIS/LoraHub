"""Cross-field consistency rules for the diffusion-pipe backend.

Same shape as the kohya / anima_lora policies — emit
``ValidationIssue`` records for combinations of dp-specific fields
that train.py asserts against at startup, or that silently degrade
the run.

dp's surface area is bigger than kohya's because it covers a long
list of arches (Flux, SDXL, SD3, Wan, HunyuanVideo, Lumina, Cosmos,
HiDream, Qwen-Image, Chroma, …). The rules below stick to checks
that hold across **every** arch dp supports; arch-specific
constraints (e.g. video-only knobs only meaningful on Wan /
HunyuanVideo) are intentionally left to the compiler so a wrong
combination still raises a clear ``CompilationError``.
"""

from __future__ import annotations

from collections.abc import Iterable

from lorahub.core.backends.base import Severity, ValidationIssue
from lorahub.core.config.schema import TrainingConfig

# Which arches in our schema are video models. dp's pipeline parallel
# + caching paths behave differently on these (per-clip latents, not
# per-image), so a couple of rules below scope themselves.
_VIDEO_ARCHES = frozenset({"hunyuan_video", "wan", "ltx_video", "cosmos"})


def check_cross_field_conflicts(cfg: TrainingConfig) -> list[ValidationIssue]:
    """Return every cross-field issue ``cfg`` triggers under dp.

    Skipped silently when ``cfg.backend.type`` isn't ``diffusion-pipe``
    — these rules read fields scoped to dp's TOML and would be
    misleading on anima / kohya configs.
    """
    if (
        cfg.backend is not None
        and cfg.backend.type
        and cfg.backend.type not in ("diffusion-pipe", "diffusion_pipe")
    ):
        return []

    issues: list[ValidationIssue] = []

    issues.extend(_pipeline_conflicts(cfg))
    issues.extend(_eval_conflicts(cfg))
    issues.extend(_dtype_conflicts(cfg))
    issues.extend(_caching_conflicts(cfg))
    issues.extend(_dataset_conflicts(cfg))
    issues.extend(_network_conflicts(cfg))
    return issues


# ---------------------------------------------------------------------- #
# Pipeline parallel + offload + compile
# ---------------------------------------------------------------------- #


def _pipeline_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    opts = cfg.backend.diffusion_pipe if cfg.backend else None
    if opts is None:
        return

    # pipeline_stages > 1 (pipeline-parallel) requires reentrant grad
    # ckpt — DeepSpeed's PP scheduler can't deal with the new
    # non-reentrant checkpoint API. Upstream README is explicit.
    if opts.pipeline_stages > 1 and not opts.reentrant_activation_checkpointing:
        yield ValidationIssue(
            Severity.error,
            "backend.diffusionPipe.reentrantActivationCheckpointing",
            f"pipelineStages={opts.pipeline_stages} 启用了 pipeline parallel,"
            "DeepSpeed PP 调度器要求 reentrantActivationCheckpointing=true。"
            "(upstream README 显式说明)。",
        )

    # blocks_to_swap 与 compile=true 同时开会让 cudagraph 反复重抓 swap
    # 进出的图,几乎不可能稳定。dp 文档建议二选一。
    blocks_to_swap = cfg.optimization.blocks_to_swap or opts.blocks_to_swap
    if blocks_to_swap > 0 and opts.compile:
        yield ValidationIssue(
            Severity.error,
            "backend.diffusionPipe.compile",
            f"blocksToSwap={blocks_to_swap} 与 compile=true 互斥。"
            "torch.compile 的 cudagraph trace 与每步换 swap 进出的 transformer "
            "块冲突,实际跑会反复重 trace 或直接崩溃。二选一。",
        )

    if opts.partition_method == "manual":
        if opts.pipeline_stages < 2:
            yield ValidationIssue(
                Severity.error,
                "backend.diffusionPipe.pipelineStages",
                "partitionMethod='manual' 需要至少两个 pipeline stage。",
            )
        elif opts.partition_split is None:
            yield ValidationIssue(
                Severity.error,
                "backend.diffusionPipe.partitionSplit",
                "partitionMethod='manual' 时必须填写 partitionSplit。",
            )
        else:
            expected = opts.pipeline_stages - 1
            if len(opts.partition_split) != expected:
                yield ValidationIssue(
                    Severity.error,
                    "backend.diffusionPipe.partitionSplit",
                    f"partitionSplit 长度 {len(opts.partition_split)} 与 pipelineStages-1={expected} 不一致。"
                    "DeepSpeed 启动会 assert,训练直接退出。",
                )
    elif opts.partition_split is not None:
        yield ValidationIssue(
            Severity.error,
            "backend.diffusionPipe.partitionSplit",
            f"partitionMethod={opts.partition_method!r} 不读取 partitionSplit。"
            "清空该字段，或将分片方式改为 manual。",
        )


# ---------------------------------------------------------------------- #
# Eval cadence
# ---------------------------------------------------------------------- #


def _eval_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    opts = cfg.backend.diffusion_pipe if cfg.backend else None
    if opts is None:
        return

    triggers = sum(
        1
        for v in (opts.eval_every_n_epochs, opts.eval_every_n_steps, opts.eval_every_n_examples)
        if v
    )
    if triggers > 1:
        yield ValidationIssue(
            Severity.warning,
            "backend.diffusionPipe.evalEveryNEpochs",
            "evalEveryN{Epochs|Steps|Examples} 同时设了多个,dp 只取最先识别到的那个,"
            "其余被忽略。建议只留一个。",
        )

    # eval_micro_batch_size_per_gpu 默认 1,但用户填了 image-only 的
    # ``image_eval_micro_batch_size_per_gpu`` 而没改一般版,会导致 video
    # 评估走 batch=1。
    if (
        opts.image_eval_micro_batch_size_per_gpu
        and opts.eval_micro_batch_size_per_gpu == 1
        and cfg.base_model.arch in _VIDEO_ARCHES
    ):
        yield ValidationIssue(
            Severity.info,
            "backend.diffusionPipe.evalMicroBatchSizePerGpu",
            "imageEvalMicroBatchSizePerGpu 已设但 evalMicroBatchSizePerGpu 仍是 1;"
            f"video arch {cfg.base_model.arch} eval 时会回到 batch=1,可能跑得很慢。",
        )

    # eval_before_first_step 但没配 eval cadence
    if opts.eval_before_first_step and triggers == 0:
        yield ValidationIssue(
            Severity.info,
            "backend.diffusionPipe.evalBeforeFirstStep",
            "evalBeforeFirstStep=true 但没设任何 evalEveryN* 周期,只会在第一步前评估一次,"
            "之后整段训练没有评估指标。",
        )


# ---------------------------------------------------------------------- #
# Dtype combinations
# ---------------------------------------------------------------------- #


def _dtype_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    opts = cfg.backend.diffusion_pipe if cfg.backend else None
    if opts is None:
        return

    transformer_dtype = opts.transformer_dtype
    diffusion_dtype = opts.diffusion_model_dtype
    precision = cfg.precision

    # fp8 transformer 与 fp32 全局精度同时存在几乎肯定不是用户本意。
    if transformer_dtype and "float8" in transformer_dtype and precision == "fp32":
        yield ValidationIssue(
            Severity.warning,
            "backend.diffusionPipe.transformerDtype",
            f"transformerDtype='{transformer_dtype}' (fp8) 但全局 precision='fp32',"
            "组合不常见 — fp8 转换在 fp32 训练里不省显存反而拖速。检查这是否有意。",
        )

    # 两个 dtype 字段同时被设,语义会冲突
    if transformer_dtype and diffusion_dtype and transformer_dtype != diffusion_dtype:
        yield ValidationIssue(
            Severity.warning,
            "backend.diffusionPipe.diffusionModelDtype",
            f"transformerDtype='{transformer_dtype}' 与 diffusionModelDtype='{diffusion_dtype}' "
            "都被设了且值不同。dp 会以 transformerDtype 为准,diffusionModelDtype 被忽略。",
        )


# ---------------------------------------------------------------------- #
# Caching
# ---------------------------------------------------------------------- #


def _caching_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    opts = cfg.backend.diffusion_pipe if cfg.backend else None
    if opts is None:
        return

    if opts.caching_batch_size > 8:
        yield ValidationIssue(
            Severity.warning,
            "backend.diffusionPipe.cachingBatchSize",
            f"cachingBatchSize={opts.caching_batch_size} 偏大,VAE / TE encode 阶段单次"
            "占的显存呈线性增长。8GB 卡上 1-2 是合理上限,16GB 4 已经偏高。",
        )

    if opts.cache_shuffle_num > 0 and opts.cache_shuffle_num < 8:
        yield ValidationIssue(
            Severity.info,
            "backend.diffusionPipe.cacheShuffleNum",
            f"cacheShuffleNum={opts.cache_shuffle_num} 太小 (< 8),shuffle 等同于不开。"
            "要么设 0(关),要么 ≥ 16。",
        )


# ---------------------------------------------------------------------- #
# Dataset / bucket
# ---------------------------------------------------------------------- #


def _dataset_conflicts(cfg: TrainingConfig) -> Iterable[ValidationIssue]:
    opts = cfg.backend.diffusion_pipe if cfg.backend else None
    if opts is None:
        return

    if cfg.dataset.reg_source is not None:
        yield ValidationIssue(
            Severity.error,
            "dataset.regSource",
            "diffusion-pipe 不读取 regSource；请将需要训练的数据集作为目录 Subsets 配置。",
        )
    if cfg.dataset.conditioning_dir is not None or any(
        subset.conditioning_data_dir is not None for subset in cfg.dataset.subsets
    ):
        yield ValidationIssue(
            Severity.error,
            "dataset.subsets",
            "diffusion-pipe 不支持 conditioningDir / conditioningDataDir；这些字段不会进入生成的 TOML。",
        )
    if cfg.dataset.val_split > 0:
        yield ValidationIssue(
            Severity.error,
            "dataset.valSplit",
            "diffusion-pipe 不会从 valSplit 自动拆分验证集；请配置独立 evalDatasets 与 evalEveryN*，"
            "或将 valSplit 设为 0。",
        )

    # min_ar / max_ar inversion
    if opts.min_ar > 0 and opts.max_ar > 0 and opts.min_ar > opts.max_ar:
        yield ValidationIssue(
            Severity.error,
            "backend.diffusionPipe.maxAr",
            f"maxAr={opts.max_ar} 小于 minAr={opts.min_ar},aspect-ratio bucket 区间为空。"
            "数据集会被全部丢弃。",
        )

    # caption shuffle / drop on caption side同 anima/kohya
    cap = cfg.dataset.caption if cfg.dataset else None
    if cap is not None:
        keep = cap.keep_tokens
        if keep is not None and keep > 0 and not cap.shuffle:
            yield ValidationIssue(
                Severity.info,
                "dataset.caption.keepTokens",
                f"dataset.caption.keepTokens={keep} 但 shuffle=false,这个值不起作用。",
            )

    # uncond_fraction 太高会摧毁 caption 信号
    if opts.uncond_fraction >= 0.5:
        yield ValidationIssue(
            Severity.warning,
            "backend.diffusionPipe.uncondFraction",
            f"uncondFraction={opts.uncond_fraction} 过高 — 一半训练步丢弃文本条件,"
            "模型会大幅偏向无条件分布。常用值 0.05-0.15。",
        )


# ---------------------------------------------------------------------- #
# Network rank consistency
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
            f"network.rank={rank} 必须 >= 1。",
        )
        return

    ratio = alpha / max(rank, 1)
    if ratio >= 4.0 or ratio <= 0.25:
        yield ValidationIssue(
            Severity.warning,
            "network.alpha",
            f"network.alpha={alpha} 与 rank={rank} 比值 {ratio:.2f} 偏离推荐区间 "
            f"[0.25, 4.0]。effective LR ∝ alpha/rank。",
        )


__all__ = ["check_cross_field_conflicts"]
