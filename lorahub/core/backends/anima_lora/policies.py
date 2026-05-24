"""Cross-field consistency rules for the anima_lora backend.

Surface every "this combination of knobs will fail or silently degrade"
case as a structured ``ValidationIssue`` instead of letting it crash
mid-launch. The compiler already raises / warns about a few of these
on the way to producing argv (see compiler._enforce_compile_constraints
and _check_ema_compile_conflict), but those paths only fire after the
caller decides to launch — too late for the UI to flag them in the
form. This module evaluates the same rules against an
``AnimaLoraOptions`` snapshot without producing argv, returning issues
the form / CLI / API can render before training starts.

The rules read static knobs only — no I/O, no GPU probe, no
side-effects — so this module is safe to import from settings UIs
that need sub-second feedback while the user types.
"""

from __future__ import annotations

import shutil
import sys
from typing import Iterable

from lorahub.core.backends.base import Severity, ValidationIssue
from lorahub.core.config.schema import AnimaLoraOptions, TrainingConfig


def check_cross_field_conflicts(cfg: TrainingConfig) -> list[ValidationIssue]:
    """Return every cross-field issue ``cfg`` triggers.

    Issues are emitted in priority order — errors first within each
    rule cluster — so a UI that only renders the first few still
    surfaces the most damaging combinations.

    Skipped silently when ``cfg.backend.type`` isn't ``anima_lora``
    — these rules assume the AnimaLoraOptions section is populated
    and would be irrelevant on kohya / diffusion-pipe configs.
    """
    if cfg.backend is not None and cfg.backend.type and cfg.backend.type != "anima_lora":
        return []

    issues: list[ValidationIssue] = []
    opts = cfg.backend.anima_lora
    if opts is None:
        return issues  # non-anima backend; nothing to do

    issues.extend(_compile_conflicts(opts))
    issues.extend(_offload_conflicts(opts))
    issues.extend(_optimizer_conflicts(cfg, opts))
    issues.extend(_network_conflicts(cfg, opts))
    issues.extend(_schedule_conflicts(cfg, opts))
    issues.extend(_validation_conflicts(opts))
    issues.extend(_caption_conflicts(cfg, opts))
    return issues


# ----------------------------------------------------------------------- #
# compile-mode conflicts
# ----------------------------------------------------------------------- #


def _compile_conflicts(opts: AnimaLoraOptions) -> Iterable[ValidationIssue]:
    """``compile_mode`` + memory-saving knobs interact poorly.

    The compiler enforces a stricter subset of these as hard errors at
    launch time (raises ``CompilationError``); we re-emit them here as
    structured issues so the form sees them before submit. We also
    cover the EMA + cudagraph_trees pair which the compiler currently
    auto-rewrites with a warning — better to nag the user up-front so
    they explicitly choose the trade-off rather than wonder why their
    requested ``reduce-overhead`` got silently downgraded.
    """
    if opts.compile_mode == "full":
        offenders: list[str] = []
        if opts.gradient_checkpointing:
            offenders.append("gradient_checkpointing=true")
        if opts.unsloth_offload_checkpointing:
            offenders.append("unsloth_offload_checkpointing=true")
        if opts.blocks_to_swap > 0:
            offenders.append(f"blocks_to_swap={opts.blocks_to_swap}")
        if offenders:
            yield ValidationIssue(
                Severity.error,
                "backend.animaLora.compileMode",
                "compile_mode='full' 与下列字段不兼容: "
                + ", ".join(offenders)
                + " — 上游 train.py 启动时会 assert。请把 compileMode 切到 "
                "'blocks' 或留空,或关掉对应的 offload 字段。",
            )

    # EMA + cudagraph_trees:仅当 inductor mode 是 reduce-overhead 时,
    # cudagraph_trees 才会启用 liveness check;EMA 的 detach/copy 必触发。
    inductor_mode = (opts.compile_inductor_mode or "").lower()
    if opts.ema and inductor_mode == "reduce-overhead":
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.ema",
            "ema=true 与 compileInductorMode='reduce-overhead' 一起会触发 "
            "cudagraph_trees liveness 检查失败 (EMA 用 detach/copy 改 LoRA 参数)。"
            "编译期会自动降级 inductor mode 到 'default' 并打 warning,但建议你"
            "显式把 compileInductorMode 设成 'default' 让意图清晰。",
        )

    # 退化路径检测:torch_compile=true + reduce-overhead 但 compile_mode 留空
    # → train.py 走的是普通 inductor (每个 bucket 形状一份 graph、无 cudagraph
    # replay),并不会用上 ``compile_blocks`` 那条 CUDA Graphs 快路径。这就是
    # 上游 fork 文章描述的 "2x 速度差距" 的诱因。仅当显存路径关掉(没 grad
    # ckpt / 没 blocks_to_swap)时才提醒,避免 8gb 这类必须走显存路径的配置
    # 被误警告。
    if (
        opts.torch_compile
        and inductor_mode == "reduce-overhead"
        and opts.compile_mode is None
        and not opts.gradient_checkpointing
        and not opts.unsloth_offload_checkpointing
        and not opts.cpu_offload_checkpointing
        and opts.blocks_to_swap == 0
    ):
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.compileMode",
            "compileInductorMode='reduce-overhead' 但 compileMode 留空 — "
            "train.py 不会调用 compile_blocks 快路径,等价于普通 inductor + "
            "每形状一份 graph、无 CUDA Graph replay。在高吞吐 GPU(RTX Pro "
            "6000 / 4090 等)上会损失约 2x 训练速度。建议显式设置 "
            "compileMode='blocks' 拿回 CUDA Graphs 加速;若是显存吃紧场景请"
            "把 compileInductorMode 改成 'default' 让意图清晰。",
        )

    # bucket_table='1536' 与 static_token_count<9240 不兼容
    # 9216+9240 双族,4096 cap 装不下任何 1536 entry。
    if (
        opts.bucket_table == "1536"
        and not opts.enable_native_flatten
        and (opts.static_token_count or 0) < 9240
    ):
        yield ValidationIssue(
            Severity.error,
            "backend.animaLora.bucketTable",
            f"bucketTable='1536' 配合 staticTokenCount={opts.static_token_count} 装不下 9240 token 的最大 entry。"
            "推荐方案:开启 enableNativeFlatten=true(zero-pad 训练 1536²);"
            "或保持静态 padding 但把 staticTokenCount 提到 9240 及以上"
            "(显存占用接近翻倍,不建议)。",
        )

    # native_flatten 与 static_token_count 互斥
    # vendored 的 compile_blocks 在两者都设置时会 raise,这里前置校验。
    if opts.enable_native_flatten and (opts.static_token_count or 0) > 0:
        yield ValidationIssue(
            Severity.error,
            "backend.animaLora.enableNativeFlatten",
            "enableNativeFlatten=true 与 staticTokenCount 互斥。"
            "native-flatten 走 4032+4200 双家族 bucket 表(每个 bucket 精确"
            "填满 token count、零 padding),staticTokenCount 走 4096 padding "
            "路径。请把 staticTokenCount 删掉(或设为 0),或关掉 native-flatten。",
        )

    # native_flatten + reduce-overhead + 显存 swap/grad-ckpt 不兼容
    # cudagraph_trees 不能跨 block swap 边界稳定捕获,且 grad ckpt 重新 forward
    # 会撞上 cudagraph slot 复用问题(我们已有 do_sample 修复)。
    if (
        opts.enable_native_flatten
        and inductor_mode == "reduce-overhead"
        and (
            opts.gradient_checkpointing
            or opts.unsloth_offload_checkpointing
            or opts.cpu_offload_checkpointing
            or opts.blocks_to_swap > 0
        )
    ):
        offenders: list[str] = []
        if opts.gradient_checkpointing:
            offenders.append("gradient_checkpointing")
        if opts.unsloth_offload_checkpointing:
            offenders.append("unsloth_offload_checkpointing")
        if opts.cpu_offload_checkpointing:
            offenders.append("cpu_offload_checkpointing")
        if opts.blocks_to_swap > 0:
            offenders.append(f"blocks_to_swap={opts.blocks_to_swap}")
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.enableNativeFlatten",
            "enableNativeFlatten=true + compileInductorMode='reduce-overhead' + "
            + ", ".join(offenders)
            + " — CUDA Graphs 与 block swap / grad ckpt 互斥(cudagraph slot "
            "无法跨 swap 边界稳定捕获)。建议把 compileInductorMode 改为 "
            "'default',或关掉 swap / 显存优化字段以拿回完整 reduce-overhead 加速。",
        )


# ----------------------------------------------------------------------- #
# offload conflicts
# ----------------------------------------------------------------------- #


def _offload_conflicts(opts: AnimaLoraOptions) -> Iterable[ValidationIssue]:
    if opts.blocks_to_swap > 0 and opts.cpu_offload_checkpointing:
        yield ValidationIssue(
            Severity.error,
            "backend.animaLora.cpuOffloadCheckpointing",
            f"blocks_to_swap={opts.blocks_to_swap} 与 "
            "cpu_offload_checkpointing=true 互斥 (anima_lora train.py:326 "
            "AssertionError)。建议保留 blocks_to_swap (省显存收益更大),把 "
            "cpu_offload_checkpointing 关掉;unsloth_offload_checkpointing "
            "可以与 blocks_to_swap 共存。",
        )

    if opts.gradient_checkpointing and opts.unsloth_offload_checkpointing:
        # 不致命,但 unsloth 的 offload 自带 checkpoint 语义,叠加会浪费显存。
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.unslothOffloadCheckpointing",
            "gradient_checkpointing 与 unsloth_offload_checkpointing 都开会"
            "做两次重算,反向时间近乎翻倍但不省额外显存。建议只开一个。",
        )


# ----------------------------------------------------------------------- #
# optimizer conflicts
# ----------------------------------------------------------------------- #


def _optimizer_conflicts(
    cfg: TrainingConfig, opts: AnimaLoraOptions,
) -> Iterable[ValidationIssue]:
    """8bit optimizers need bitsandbytes; LR / scheduler shape sanity."""
    optimizer_type = (opts.optimizer_type or cfg.optimizer.type or "").strip()
    is_8bit = optimizer_type.lower().endswith("8bit")
    if is_8bit and not _has_bitsandbytes():
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.optimizerType",
            f"optimizer_type='{optimizer_type}' 需要 bitsandbytes,但当前活跃 venv "
            "里检测不到这个包。在 anima_lora 子环境里可能有,但若运行时也找不到,"
            "训练会以 'CUDA Setup failed despite GPU being available' 退出。"
            "在 anima_lora 的 .venv 里 `uv pip install bitsandbytes` 可以装上。",
        )

    # network_train_unet_only=true 时 text-encoder LR 完全不生效。
    # 只在用户主动改过(与 unet lr 不同, 或者非默认 5e-5)时给提示——
    # 否则全部 anima 模板都会触发噪声 info。
    te_lr = cfg.optimizer.lr.text_encoder
    unet_lr = cfg.optimizer.lr.unet
    user_set_te = (
        te_lr is not None
        and te_lr != 5.0e-5  # the schema default
        and te_lr != unet_lr  # mirroring unet lr is the common idiom
    )
    if opts.network_train_unet_only and user_set_te:
        yield ValidationIssue(
            Severity.info,
            "optimizer.lr.textEncoder",
            f"networkTrainUnetOnly=true 时 textEncoder LR ({te_lr}) 不会生效 "
            "(只训练 UNet,文本编码器冻结)。如果不想让用户误以为这个值生效,"
            "建议把 textEncoder LR 与 unet LR 设成一样,或留 schema 默认 5e-5。",
        )

    # discreteFlowShift 与 weightingScheme=min_snr_rf 通常成对出现:开了 min_snr
    # 没设 minSnrGamma 是无效配置。
    if opts.weighting_scheme == "min_snr_rf" and not opts.min_snr_gamma:
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.minSnrGamma",
            "weightingScheme='min_snr_rf' 需要 minSnrGamma > 0,当前未设置 "
            "(默认 0)等价于关闭 min-SNR 重加权。建议显式设 5.0 (常用值)。",
        )


def _has_bitsandbytes() -> bool:
    """Coarse probe: just check if importable from this venv.

    In production lorahub launches the trainer in the anima_lora
    sub-venv (not the API venv), so this is a *hint* — false-positive
    on a clean API venv that delegates training to a properly-equipped
    sub-venv. We still surface it as a warning, never an error, so the
    user only sees a friendly nudge instead of a hard block.
    """
    try:
        import importlib.util  # noqa: PLC0415

        return importlib.util.find_spec("bitsandbytes") is not None
    except Exception:  # noqa: BLE001
        return False


# ----------------------------------------------------------------------- #
# network / rank consistency
# ----------------------------------------------------------------------- #


def _network_conflicts(
    cfg: TrainingConfig, opts: AnimaLoraOptions,
) -> Iterable[ValidationIssue]:
    """LoRA rank / alpha / min_rank ratio shouldn't fight each other."""
    sub = opts.lora
    if sub is None:
        return

    if sub.min_rank > opts.network_dim:
        yield ValidationIssue(
            Severity.error,
            "backend.animaLora.lora.minRank",
            f"lora.minRank={sub.min_rank} 大于 networkDim={opts.network_dim}, "
            "T-LoRA 路由层无法构造 (timestep mask 必须在 [minRank, networkDim] "
            "区间内取整)。请把 minRank 调到 ≤ networkDim,或者把 networkDim 拉高。",
        )

    # alpha == dim 是上游推荐的 OrthoLoRA 默认;偏离 4× 以上几乎肯定是手抖。
    ratio = opts.network_alpha / max(opts.network_dim, 1)
    if ratio >= 4.0 or ratio <= 0.25:
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.networkAlpha",
            f"networkAlpha={opts.network_alpha} 与 networkDim={opts.network_dim} "
            f"之比 {ratio:.2f} 偏离推荐区间 [0.25, 4.0]。LoRA 数学上 effective LR "
            "= alpha/rank * base_lr,过大会数值不稳,过小近乎不学。建议 alpha 与 "
            "rank 相等,或在 0.5×~2× 之间。",
        )

    # network rank=1 是 schema 允许的下界,但训练几乎学不到东西
    if opts.network_dim < 4:
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.networkDim",
            f"networkDim={opts.network_dim} 过低,LoRA 几乎学不到内容。"
            "推荐 8-32 之间;character LoRA 一般 16-32,style LoRA 4-16。",
        )


# ----------------------------------------------------------------------- #
# schedule consistency
# ----------------------------------------------------------------------- #


def _schedule_conflicts(
    cfg: TrainingConfig, opts: AnimaLoraOptions,
) -> Iterable[ValidationIssue]:
    """Save / checkpoint cadence vs total epochs."""
    max_epochs = opts.max_train_epochs or cfg.schedule.epochs

    if opts.save_every_n_epochs and opts.save_every_n_epochs > max_epochs:
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.saveEveryNEpochs",
            f"saveEveryNEpochs={opts.save_every_n_epochs} 大于 "
            f"maxTrainEpochs={max_epochs},整次训练只会在最后落一次盘 "
            "(没有中间 checkpoint)。建议设到 max_epochs 的 1/4 ~ 1/2。",
        )

    if opts.checkpointing_epochs and opts.checkpointing_epochs > max_epochs:
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.checkpointingEpochs",
            f"checkpointingEpochs={opts.checkpointing_epochs} 大于 "
            f"maxTrainEpochs={max_epochs},不会有中间 state 可 resume。",
        )

    # batch_size * grad_accum > dataset_size 意味着每个 epoch 最多 1 个 step
    bs = cfg.schedule.batch_size
    accum = max(cfg.schedule.grad_accum, 1)
    effective_batch = bs * accum
    # dataset size unknown at this point — use a heuristic floor.
    # Anything beyond 64 means grad signal is heavily smeared.
    if effective_batch > 64:
        yield ValidationIssue(
            Severity.warning,
            "schedule.gradAccum",
            f"effective batch (batchSize × gradAccum = {bs}×{accum} = "
            f"{effective_batch}) 偏大。LoRA 微调常见的 effective batch 在 4~16,"
            "过大会让梯度信号过度平均、收敛变慢。",
        )


# ----------------------------------------------------------------------- #
# validation / sampling
# ----------------------------------------------------------------------- #


def _validation_conflicts(opts: AnimaLoraOptions) -> Iterable[ValidationIssue]:
    # 16 是 schema 默认 (来自 anima 上游 base.toml), 不当成"用户主动开了 holdout"
    # 处理。只在用户**显式改成非默认值**且 use_cmmd=false 时给提示。
    SPLIT_DEFAULT = 16
    if (
        opts.validation_split_num
        and opts.validation_split_num > 0
        and opts.validation_split_num != SPLIT_DEFAULT
        and not opts.use_cmmd
    ):
        yield ValidationIssue(
            Severity.info,
            "backend.animaLora.useCmmd",
            f"validationSplitNum={opts.validation_split_num} 但 useCmmd=false。"
            "划出的 holdout 样本会被加载但不会产出任何指标,只是浪费显存与磁盘。"
            "需要监控就把 useCmmd 设为 true,不需要就把 validationSplitNum 设 0。",
        )


# ----------------------------------------------------------------------- #
# caption / dataset
# ----------------------------------------------------------------------- #


def _caption_conflicts(
    cfg: TrainingConfig, opts: AnimaLoraOptions,
) -> Iterable[ValidationIssue]:
    """keep_tokens 与 caption.shuffle 的相互作用。"""
    keep = opts.keep_tokens if opts.keep_tokens is not None else cfg.dataset.caption.keep_tokens
    if keep is not None and keep > 0 and not opts.use_shuffled_caption_variants:
        # keepTokens 在 shuffle off 时是 no-op (没有任何 token 会被打乱),
        # 用户多半误以为它在做 trigger word 锚定。
        yield ValidationIssue(
            Severity.info,
            "backend.animaLora.keepTokens",
            f"keepTokens={keep} 在 useShuffledCaptionVariants=false 时不起作用 "
            "(没有 shuffle 就没有要锚定的对象)。要么开 shuffle,要么把 keepTokens 设为 0。",
        )

    drop_rate = opts.caption_dropout_rate
    if drop_rate is not None and drop_rate >= 0.5:
        yield ValidationIssue(
            Severity.warning,
            "backend.animaLora.captionDropoutRate",
            f"captionDropoutRate={drop_rate} 过高,意味着每步至少一半的样本"
            "完全丢弃 caption。这会让模型迅速忘掉 trigger word。常用值在 0.05-0.15。",
        )


__all__ = ["check_cross_field_conflicts"]
