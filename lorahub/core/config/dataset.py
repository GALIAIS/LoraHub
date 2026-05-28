"""Dataset, bucket, and caption configs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from ._shared import _CAMEL_CONFIG


class BucketConfig(BaseModel):
    enabled: bool = True
    min_size: int = Field(256, alias="min")
    max_size: int = Field(2048, alias="max")
    step: int = 64
    # Don't upscale images smaller than the bucket; clamps tiny inputs
    # instead of stretching them. kohya: --bucket_no_upscale.
    no_upscale: bool = False
    # Skip the resolution sanity check (kohya: --skip_image_resolution).
    # Useful for datasets with unusual aspect ratios.
    skip_image_resolution: bool = False
    # PIL resampling kernel. None lets the trainer pick its default.
    # Mirrors kohya's --bucket_reso_steps companion flag's accepted set.
    resize_interpolation: Literal[
        "lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"
    ] | None = None
    # diffusion-pipe accepts an explicit AR list overriding min/max/num.
    # Each entry is a width/height ratio; only consumed by the dp compiler.
    ar_buckets: list[float] | None = None

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class CaptionConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    # NOTE: ``strategy`` is consumed by the front-end form (it gates
    # which UI controls render). The kohya / diffusion-pipe compilers
    # don't read it — they always look for an ``<image>.txt`` companion
    # file regardless of strategy. Keep it in the schema so configs
    # round-trip cleanly through the UI, but treat it as documentation
    # of intent rather than a backend-driving knob.
    strategy: Literal["tag_file", "filename", "none"] = "tag_file"
    ext: str = ".txt"
    shuffle: bool = True
    drop_rate: float = Field(0.0, ge=0.0, le=1.0)
    # Per-epoch caption dropout (kohya: --caption_dropout_every_n_epochs).
    # Different from drop_rate which is per-step.
    dropout_every_n_epochs: int = Field(0, ge=0)
    # Per-tag dropout — drops individual tags within a caption.
    tag_dropout_rate: float = Field(0.0, ge=0.0, le=1.0)
    # First N comma-separated tokens are NEVER shuffled away; pinning
    # the trigger word at index 0 is the typical use.
    keep_tokens: int = Field(0, ge=0)
    # Hard-drop list — every entry that appears in a caption is
    # removed verbatim before training (case-insensitive substring
    # match, then comma-list cleanup). Entries can be either tag-style
    # (``"1girl"``, ``"looking at viewer"``) or natural-language
    # phrases (``"a person standing in front of a window"``). Applied
    # at compile time to a sanitised mirror of the dataset under
    # ``<workspace>/captions_sanitized/`` so the trainer reads the
    # filtered text, but the user's source ``.txt`` files are left
    # untouched. Empty list = no-op (the mirror step is also skipped).
    drop_tokens: list[str] = Field(default_factory=list)
    # Custom separator between "kept" and shufflable tokens; default ","
    keep_tokens_separator: str | None = None
    # Secondary separator within a token group (e.g. " ,"). kohya only.
    secondary_separator: str | None = None
    # `{a|b|c}` wildcard support in captions (kohya: --enable_wildcard).
    enable_wildcard: bool = False
    # Compose-time prefix/suffix prepended/appended to every caption.
    prefix: str | None = None
    suffix: str | None = None
    # Max tokenizer length (kohya: --max_token_length, valid: 150/225;
    # 75 is the implicit default when the flag is absent so it's
    # represented as None here rather than an explicit value).
    max_token_length: Literal[150, 225] | None = None
    # Token warmup (slow-start the tag count). kohya only.
    token_warmup_min: int | None = Field(default=None, ge=1)
    token_warmup_step: float | None = Field(default=None, ge=0)
    # Weighted captions (lpw-style `(token:1.5)`). kohya: --weighted_captions.
    weighted: bool = False
    # dp-only: tag shuffle delimiter (default ", ") and legacy whole-caption shuffle.
    shuffle_delimiter: str | None = None
    shuffle_tags: bool = False


class DatasetSubsetConfig(BaseModel):
    """One [[directory]] entry on the dp side; kohya squashes these into
    `--train_data_dir` semantics via per-subset toml."""

    model_config = _CAMEL_CONFIG

    path: Path
    num_repeats: int = Field(1, ge=1)
    # Optional mask directory, mirrors the image dir layout.
    mask_path: Path | None = None
    # Per-subset bucket override (dp).
    ar_buckets: list[float] | None = None
    # Per-subset caption override.
    caption_prefix: str | None = None
    # Conditioning training (anima_lora 差异训练): same-stem reference
    # image directory paired with this subset's image_dir. Loaded into
    # ``batch['conditioning_images']`` when the trainer is launched
    # with ``--conditioning``. None disables.
    conditioning_data_dir: Path | None = None


class DatasetConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    source: Path
    resolution: list[int] = Field(default_factory=lambda: [1024, 1024])
    bucket: BucketConfig = Field(default_factory=lambda: BucketConfig())
    caption: CaptionConfig = Field(default_factory=lambda: CaptionConfig())
    num_repeats: int = Field(1, ge=1)
    # Fraction of the dataset reserved for held-out validation. `0.0` disables
    # validation entirely (the previous behaviour); upper bound stays under
    # 0.5 because anything more would be a strange split. sd-scripts' flag
    # `--validation_split_percentage` takes an integer percent — we convert
    # at compile time.
    val_split: float = Field(0.0, ge=0.0, lt=0.5)
    # Multi-directory support — when populated, OVERRIDES `source`.
    # dp emits one [[directory]] block per entry; kohya synthesises an
    # equivalent dataset toml.
    subsets: list[DatasetSubsetConfig] = Field(default_factory=list)
    # Video training: list of frame counts (e.g. `[1, 33, 65]`).
    # Default `[1]` means image-only.
    frame_buckets: list[int] = Field(default_factory=lambda: [1])
    # ControlNet / inpainting conditioning images (kohya: --conditioning_data_dir).
    conditioning_dir: Path | None = None
    # DreamBooth regularisation set (kohya: --reg_data_dir).
    reg_source: Path | None = None

    @model_validator(mode="after")
    def _validate_resolution(self) -> DatasetConfig:
        if len(self.resolution) not in (1, 2):
            msg = "resolution must be [size] or [width, height]"
            raise ValueError(msg)
        return self
