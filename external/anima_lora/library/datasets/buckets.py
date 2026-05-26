import math
import random
from typing import NamedTuple, Optional, Tuple

import numpy as np

# Bucket resolutions where (W/16)*(H/16) <= 4096 tokens, with minimal padding
# to reach 4096.  Using these with static_token_count=4096 makes every forward
# pass shape-identical, eliminating torch.compile recompilation.
# Landscape mirrors (swap W, H) are included explicitly.
CONSTANT_TOKEN_BUCKETS = [
    (1024, 1024),  # 4096 tokens, 0.0% pad
    (960, 1088),  # 4080 tokens, 0.4% pad
    (1088, 960),
    (896, 1152),  # 4032 tokens, 1.6% pad
    (1152, 896),
    (832, 1248),  # 4056 tokens, 1.0% pad
    (1248, 832),
    (768, 1344),  # 4032 tokens, 1.6% pad
    (1344, 768),
    (720, 1440),  # 4050 tokens, 1.1% pad
    (1440, 720),
    (640, 1632),  # 4080 tokens, 0.4% pad
    (1632, 640),
    (576, 1792),  # 4032 tokens, 1.6% pad
    (1792, 576),
    (512, 2048),  # 4096 tokens, 0.0% pad
    (2048, 512),
]

# Native-flatten bucket table — paired with compile_blocks(native_flatten=True)
# (see :func:`Anima.compile_blocks`). Six token-count families: 4032 (=63*64),
# 4200 (=60*70), 4096 (=64*64), 6144 (=96*64), 9216 (=96*96), and 1024 (=32*32).
# The first two are highly composite, so each factors into many
# near-square→elongated patch grids — and crucially every bucket *exactly*
# fills its token count, so there is zero intra-bucket padding by construction.
#
# The first two families densely cover aspect space; a single token count's
# divisors near √N are sparse (4032 alone jumps aspect 1.29→1.75), so
# interleaving 4032 and 4200 densely covers aspect space at the cost of one
# extra graph. The additional families (4096, 6144, 9216, 1024) provide
# exact-match buckets for common resolutions. Landscape mirrors (swap W, H)
# are included explicitly. Token count = (W//16)*(H//16). The rope
# per-axis cap is 256 patches; the largest dim here is 2016px → 126.
#
# Use this with ``BucketManager.make_buckets(native_token_buckets=True)`` (or
# pass ``CONSTANT_TOKEN_BUCKETS_NATIVE`` to ``set_predefined_resos`` directly)
# and ``compile_blocks(native_flatten=True)`` together. Without
# native_flatten, the 4200-token rows would overflow ``static_token_count=4096``;
# use the legacy 4096-cap table above for the static-pad path.
CONSTANT_TOKEN_BUCKETS_NATIVE = [
    # ---- 4032-token family (63*64) ----
    (1008, 1024),  # 63 x 64, ar 0.98 (nearest to square)
    (1024, 1008),  #          ar 1.02
    (896, 1152),  # 56 x 72, ar 0.78
    (1152, 896),  #          ar 1.29
    (768, 1344),  # 48 x 84, ar 0.57
    (1344, 768),  #          ar 1.75
    (672, 1536),  # 42 x 96, ar 0.44
    (1536, 672),  #          ar 2.29
    (576, 1792),  # 36 x 112, ar 0.32
    (1792, 576),  #           ar 3.11
    (512, 2016),  # 32 x 126, ar 0.25
    (2016, 512),  #           ar 3.94
    # ---- 4200-token family (60*70) ----
    (960, 1120),  # 60 x 70, ar 0.86
    (1120, 960),  #          ar 1.17
    (896, 1200),  # 56 x 75, ar 0.75
    (1200, 896),  #          ar 1.34
    (800, 1344),  # 50 x 84, ar 0.60
    (1344, 800),  #          ar 1.68
    (672, 1600),  # 42 x 100, ar 0.42
    (1600, 672),  #           ar 2.38
    (640, 1680),  # 40 x 105, ar 0.38
    (1680, 640),  #           ar 2.62
    (560, 1920),  # 35 x 120, ar 0.29
    (1920, 560),  #           ar 3.43
    # ---- 4096-token family (64*64) ----
    (1024, 1024),  # 64 x 64, ar 1.00 (exact square)
    # ---- 6144-token family (96*64) ----
    (1536, 1024),  # 96 x 64, ar 1.50
    (1024, 1536),  # 64 x 96, ar 0.67
    # ---- 9216-token family (96*96) ----
    (1536, 1536),  # 96 x 96, ar 1.00 (large square)
    # ---- 1024-token family (32*32) ----
    (512, 512),  # 32 x 32, ar 1.00 (small square)
]

# 1536-resolution bucket table — paired with compile_blocks(native_flatten=True)
# to train a LoRA at Anima v1.0's native 1536x1536 inference resolution.
# 9216 (=96*96) is the load-bearing token count for square 1536; we add 9240
# (=66*140, 70*132, 77*120, 84*110, 88*105 …) as the second family because
# 9216's near-square divisors are sparse (jumps 0.56→1.00 with nothing in
# between), and 9240 fills the 0.53-0.84 aspect range that real datasets
# usually live in. Both families compile to ONE block graph each (two graphs
# total, same overhead as the 4032+4200 NATIVE table).
#
# Pair this with ``BucketManager.make_buckets(table_name="1536")`` and
# ``compile_blocks(native_flatten=True)``. Static-pad mode requires
# ``static_token_count >= 9240`` to fit the largest entry; native-flatten
# is the recommended path because the rope per-axis cap of 256 patches is
# fully exercised here (the 2304x1024 entry runs into 144 patches on the
# long axis — well within rope's 256 ceiling).
#
# Throughput note: 1536² training is ~2.25x the per-step compute of 1024²
# (token count 9216 vs 4096) and the rope/attention sequence length scales
# linearly; budget for it accordingly. On RTX Pro 6000 / 4090 with
# native_flatten + reduce-overhead this still trains comfortably.
CONSTANT_TOKEN_BUCKETS_1536 = [
    # ---- 9216-token family (96*96) — square + landscape mirrors ----
    (1536, 1536),  # 96 x 96,   ar 1.00 (square — Anima v1.0 native)
    (1152, 2048),  # 72 x 128,  ar 0.56
    (2048, 1152),  #            ar 1.78
    (1024, 2304),  # 64 x 144,  ar 0.44
    (2304, 1024),  #            ar 2.25
    # ---- 9240-token family — fills the 0.53-0.84 aspect gap ----
    (1408, 1680),  # 88 x 105,  ar 0.84
    (1680, 1408),  #            ar 1.19
    (1344, 1760),  # 84 x 110,  ar 0.76
    (1760, 1344),  #            ar 1.31
    (1232, 1920),  # 77 x 120,  ar 0.64
    (1920, 1232),  #            ar 1.56
    (1120, 2112),  # 70 x 132,  ar 0.53
]

# DCW v4 calibration aspect-bucket set.
#
# Top 5 (H, W) resolutions by frequency in post_image_dataset/lora/. List
# order *is* the canonical aspect_id index — DCW v4's per-aspect statistics
# (fusion_head.safetensors per-bucket μ_g, σ²_prior, λ_scalar) key off this
# order, so a reorder invalidates every shipped fusion-head checkpoint.
#
# Read by both the calibration data-gen path (scripts/tasks/dcw.py drives
# `make dcw` over these buckets) and the fusion-head trainer
# (scripts/dcw/fusion_data.py uses the dict for the (H, W) → aspect_id
# lookup that decides which run rows feed the trainer). Inference itself
# is bucket-agnostic post-cleanup — see project_dcw_bucket_prior_cosmetic.
DCW_ASPECT_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (832, 1248),  # 0 — HD portrait (most common)
    (896, 1152),  # 1 — 3:4 portrait
    (768, 1344),  # 2 — tall portrait
    (1152, 896),  # 3 — 3:4 landscape
    (1248, 832),  # 4 — HD landscape
)
DCW_ASPECT_NAMES: Tuple[str, ...] = tuple(
    f"{h}x{w}" for h, w in DCW_ASPECT_BUCKETS
)
DCW_ASPECT_TABLE: dict = {hw: i for i, hw in enumerate(DCW_ASPECT_BUCKETS)}
N_DCW_ASPECTS: int = len(DCW_ASPECT_BUCKETS)


def make_bucket_resolutions(max_reso, min_size=256, max_size=1024, divisible=64):
    """Generate bucket resolutions for multi-aspect-ratio training.
    Moved from model_util.py to avoid dependency."""
    max_width, max_height = max_reso
    max_area = max_width * max_height

    resos = set()

    width = int(math.sqrt(max_area) // divisible) * divisible
    resos.add((width, width))

    width = min_size
    while width <= max_size:
        height = min(max_size, int((max_area // width) // divisible) * divisible)
        if height >= min_size:
            resos.add((width, height))
            resos.add((height, width))

        width += divisible

    resos = list(resos)
    resos.sort()
    return resos


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], (
                    "the max_size should be larger than the width of max_reso"
                )
                assert max_size >= max_reso[1], (
                    "the max_size should be larger than the height of max_reso"
                )
            if min_size is not None:
                assert max_size >= min_size, (
                    "the max_size should be larger than the min_size"
                )

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(
        self,
        constant_token_buckets: bool = False,
        native_token_buckets: bool = False,
        bucket_table: Optional[str] = None,
    ):
        """Pick the resolution table that drives bucketing.

        Selection priority (later args win):
          1. ``bucket_table="1536"`` → ``CONSTANT_TOKEN_BUCKETS_1536``
             (9216-token square + 9240 fill — for Anima v1.0 native
             1536² training; pair with ``compile_blocks(native_flatten=True)``).
          2. ``native_token_buckets=True`` → ``CONSTANT_TOKEN_BUCKETS_NATIVE``
             (4032+4200 two-family table for native_flatten path).
          3. ``constant_token_buckets=True`` → ``CONSTANT_TOKEN_BUCKETS``
             (legacy 4096-pad table, paired with ``static_token_count=4096``).
          4. Default → ``make_bucket_resolutions`` (multi-AR fallback).
        """
        if bucket_table == "1536":
            resos = list(CONSTANT_TOKEN_BUCKETS_1536)
        elif bucket_table is not None and bucket_table != "default":
            raise ValueError(
                f"unknown bucket_table {bucket_table!r}; expected '1536' or None"
            )
        elif native_token_buckets:
            resos = list(CONSTANT_TOKEN_BUCKETS_NATIVE)
        elif constant_token_buckets:
            resos = list(CONSTANT_TOKEN_BUCKETS)
        else:
            resos = make_bucket_resolutions(
                self.max_reso, self.min_size, self.max_size, self.reso_steps
            )
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                abs_ar_errors = np.abs(ar_errors)
                min_ar_error = abs_ar_errors.min()
                tied = np.where(abs_ar_errors == min_ar_error)[0]
                if len(tied) > 1:
                    image_area = image_width * image_height
                    areas = np.array(
                        [w * h for w, h in self.predefined_resos]
                    )[tied]
                    predefined_bucket_id = tied[
                        np.abs(areas - image_area).argmin()
                    ]
                else:
                    predefined_bucket_id = tied[0]
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (
                int(image_width * scale + 0.5),
                int(image_height * scale + 0.5),
            )
        else:
            if image_width * image_height > self.max_area:
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, (
                    "aspect is illegal"
                )

                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                if abs(ar_width_rounded - aspect_ratio) < abs(
                    ar_height_rounded - aspect_ratio
                ):
                    resized_size = (
                        b_width_rounded,
                        int(b_width_rounded / aspect_ratio + 0.5),
                    )
                else:
                    resized_size = (
                        int(b_height_rounded * aspect_ratio + 0.5),
                        b_height_rounded,
                    )
            else:
                resized_size = (image_width, image_height)

            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = (bucket_reso[0] - resized_width) // 2
        crop_top = (bucket_reso[1] - resized_height) // 2
        crop_right = crop_left + resized_width
        crop_bottom = crop_top + resized_height
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int
