"""Bilingual help text for config fields and LoRA variant descriptions.

Per-field tooltips (short, dense, frequently looked up) live inline as
``FIELD_HELP`` / ``PREPROCESS_FIELD_HELP``. The bulky method/variant guide
HTML blocks live under ``guides/<name>.<lang>.html`` — one file per method
per language — and are loaded lazily on first access. Shared snippets
(``_apply_note``, ``_not_mergeable``) follow the same convention with an
underscore prefix.
"""

from __future__ import annotations

import functools
from pathlib import Path

from gui.i18n import current_language

_GUIDES_DIR = Path(__file__).parent / "guides"


@functools.lru_cache(maxsize=None)
def _read_guide(name: str, lang: str) -> str:
    path = _GUIDES_DIR / f"{name}.{lang}.html"
    if not path.exists():
        path = _GUIDES_DIR / f"{name}.en.html"
    return path.read_text(encoding="utf-8")


def _guide(name: str) -> str:
    return _read_guide(name, current_language())


# ── Per-field tooltips ─────────────────────────────────────────
# Keys match config field names. Each maps to {lang: description}.

FIELD_HELP: dict[str, dict[str, str]] = {
    # Architecture
    "network_dim": {
        "en": "LoRA rank (dimension of low-rank matrices). Higher = more expressive but more VRAM. Typical: 8–64.",
        "ko": "LoRA 랭크 (저랭크 행렬의 차원). 높을수록 표현력이 좋지만 VRAM 사용량 증가. 일반적: 8–64.",
    },
    "network_alpha": {
        "en": "LoRA scaling factor. Effective scale = alpha / dim. When alpha == dim, scale is 1.0. Lower alpha = more conservative updates.",
        "ko": "LoRA 스케일링 계수. 실효 스케일 = alpha / dim. alpha == dim이면 1.0. 낮을수록 보수적 업데이트.",
    },
    "network_module": {
        "en": "Python module path for the LoRA network implementation.",
        "ko": "LoRA 네트워크 구현의 Python 모듈 경로.",
    },
    "use_timestep_mask": {
        "en": "Enable T-LoRA: effective rank varies with denoising timestep via power-law schedule. Full rank at high noise, reduced at low noise.",
        "ko": "T-LoRA 활성화: 디노이징 타임스텝에 따라 유효 랭크 변동. 높은 노이즈에서 전체 랭크, 낮은 노이즈에서 축소.",
    },
    "use_ortho": {
        "en": "Enable OrthoLoRA: SVD-based orthogonal parameterization of the update matrix (linear layers only). Regularizes toward structured updates; saved as plain LoRA via thin SVD at checkpoint time.",
        "ko": "OrthoLoRA 활성화: 업데이트 행렬의 SVD 기반 직교 파라미터화 (선형 레이어 전용). 구조화된 업데이트로 정규화되며, 저장 시 thin SVD로 일반 LoRA로 변환.",
    },
    "use_moe_style": {
        "en": "MoE expert layout: 'shared_A' (HydraLoRA — one shared lora_down + N per-expert lora_up heads), 'independent_A' (FeRA — N fully-independent down/up pairs), or false (no MoE). Produces a *_moe.safetensors sibling for router-live inference; requires cache_llm_adapter_outputs=true.",
        "ko": "MoE 전문가 레이아웃: 'shared_A' (HydraLoRA — 공유 lora_down + N개 전문가별 lora_up), 'independent_A' (FeRA — 독립적인 N쌍의 down/up), 또는 false (MoE 비활성화). 라우터-라이브 추론용 *_moe.safetensors 동반 파일 생성. cache_llm_adapter_outputs=true 필요.",
    },
    "route_per_layer": {
        "en": "If true, each layer owns its own router (per-layer routing). If false, a single network-level GlobalRouter broadcasts gate weights to every routed module.",
        "ko": "true이면 레이어별 라우터 사용 (per-layer routing). false이면 네트워크 전역 GlobalRouter 하나가 모든 라우팅 모듈에 게이트 가중치 브로드캐스트.",
    },
    "router_source": {
        "en": "Routing signal: 'sigma' (sinusoidal embedding of the denoising timestep), 'fei' (mean-pooled rank features from preceding LoRA modules), or 'pooled_text' (pooled T5 caption embedding).",
        "ko": "라우팅 신호: 'sigma' (디노이징 타임스텝의 sinusoidal 임베딩), 'fei' (선행 LoRA 모듈의 평균 풀링 랭크 특징), 또는 'pooled_text' (T5 캡션 풀링 임베딩).",
    },
    "num_experts": {
        "en": "HydraLoRA expert count. More experts = more capacity but more VRAM and slower training. Typical: 2–8.",
        "ko": "HydraLoRA 전문가 수. 많을수록 표현력 증가하지만 VRAM 사용량 증가 및 학습 속도 감소. 일반적: 2–8.",
    },
    "balance_loss_weight": {
        "en": "HydraLoRA load-balancing loss weight. Discourages router collapse onto a single expert. Typical: 0.01.",
        "ko": "HydraLoRA 부하 균형 손실 가중치. 라우터가 단일 전문가로 붕괴되는 것을 방지. 일반적: 0.01.",
    },
    "balance_loss_warmup_ratio": {
        "en": "Fraction of training steps to hold the balance loss at 0 before activating it. Lets the router specialize first, then switches the penalty on to stop further collapse of a diverged router. 0.0 disables the warmup. Typical: 0.3–0.5.",
        "ko": "밸런스 손실을 0으로 유지하는 학습 스텝 비율. 먼저 라우터가 전문화되도록 한 뒤 페널티를 활성화해 분화된 라우터의 추가 붕괴를 방지. 0.0 = 비활성화. 일반적: 0.3–0.5.",
    },
    "add_reft": {
        "en": "Enable ReFT: block-level residual-stream intervention (Wu et al. 2024). Adds R^T·(ΔW·h + b)·scale to each selected DiT block's output. Composes with any LoRA variant.",
        "ko": "ReFT 활성화: 블록 수준 잔차 스트림 개입 (Wu et al. 2024). 선택된 DiT 블록 출력에 R^T·(ΔW·h + b)·scale 추가. 모든 LoRA 변형과 함께 사용 가능.",
    },
    "reft_dim": {
        "en": "ReFT intervention rank — dimension of R and ΔW in each ReFTModule. Typical: 32–64.",
        "ko": "ReFT 개입 랭크 — 각 ReFTModule의 R 및 ΔW 차원. 일반적: 32–64.",
    },
    "reft_alpha": {
        "en": "ReFT scaling factor (effective scale = alpha / dim). Typical: same as reft_dim.",
        "ko": "ReFT 스케일링 계수 (실효 스케일 = alpha / dim). 일반적: reft_dim과 동일.",
    },
    "reft_layers": {
        "en": "Which DiT blocks receive ReFT modules. 'all', 'last_8', 'first_4', 'stride_2', or comma-separated indices like '3,7,11,15'.",
        "ko": "ReFT 모듈이 적용될 DiT 블록. 'all', 'last_8', 'first_4', 'stride_2', 또는 '3,7,11,15'와 같은 쉼표 구분 인덱스.",
    },
    "sigma_feature_dim": {
        "en": "Sinusoidal σ feature dimension fed into the σ-router bias MLP. Typical: 128.",
        "ko": "σ 라우터 바이어스 MLP에 입력되는 sinusoidal σ 특징 차원. 일반적: 128.",
    },
    "router_targets": {
        "en": "Regex over layer names — only matching Linears participate in routed adaptation (Hydra MoE leaves + σ / FEI feature concatenation share the same scope). Typical: '.*(mlp\\.layer[12])$' to confine MoE to the FFN sublayers.",
        "ko": "레이어 이름에 대한 정규식 — 일치하는 Linear만 라우팅된 적응에 참여 (Hydra MoE leaves + σ / FEI 특징 연결이 동일한 범위를 공유). 일반적: '.*(mlp\\.layer[12])$' — FFN 서브레이어로 MoE 제한.",
    },
    "per_bucket_balance_weight": {
        "en": "Extra per-σ-bucket load-balance penalty, scaled by balance_loss_weight. Encourages routing diversity within each timestep bucket. Typical: 0.3.",
        "ko": "σ 버킷별 추가 부하 균형 페널티, balance_loss_weight로 스케일. 각 타임스텝 버킷 내 라우팅 다양성 유도. 일반적: 0.3.",
    },
    "num_sigma_buckets": {
        "en": "Number of timestep buckets used for per-bucket balance accounting. Typical: 3 (low / mid / high noise).",
        "ko": "버킷별 균형 계산에 사용되는 타임스텝 버킷 수. 일반적: 3 (저/중/고 노이즈).",
    },
    "specialize_experts_by_sigma_buckets": {
        "en": "Hard-partition the expert pool into σ-bands: each timestep bucket only routes to its assigned experts. Forces specialization on top of the soft σ-router bias. Pairs with sigma_bucket_boundaries.",
        "ko": "전문가 풀을 σ-밴드로 하드 분할: 각 타임스텝 버킷은 할당된 전문가만 사용. 소프트 σ-라우터 바이어스 위에 강제 특화 부여. sigma_bucket_boundaries와 함께 사용.",
    },
    "sigma_bucket_boundaries": {
        "en": "Custom σ-bucket edges, length = num_sigma_buckets + 1, monotone 0.0 → 1.0. Defaults to uniform linspace(0, 1, N+1) when omitted. Example: [0.0, 0.5, 0.8, 1.0].",
        "ko": "사용자 지정 σ-버킷 경계, 길이 = num_sigma_buckets + 1, 0.0 → 1.0 단조 증가. 생략 시 uniform linspace(0, 1, N+1) 사용. 예: [0.0, 0.5, 0.8, 1.0].",
    },
    "network_args": {
        "en": "Extra kwargs passed to the network module. For postfix: list of 'key=value' strings (e.g., 'mode=cond', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Pick a Variant to auto-fill.",
        "ko": "네트워크 모듈에 전달되는 추가 kwargs. postfix의 경우 'key=value' 문자열 리스트 (예: 'mode=cond', 'splice_position=end_of_sequence', 'cond_hidden_dim=256'). Variant 선택으로 자동 채우기 가능.",
    },
    "min_rank": {
        "en": "Minimum active rank when T-LoRA timestep masking is enabled. At the lowest-noise timesteps, rank drops to this value.",
        "ko": "T-LoRA 타임스텝 마스킹 사용 시 최소 활성 랭크. 가장 낮은 노이즈에서 이 값까지 감소.",
    },
    "alpha_rank_scale": {
        "en": "Scale alpha proportionally when T-LoRA reduces rank, keeping effective learning rate stable across timesteps.",
        "ko": "T-LoRA가 랭크를 줄일 때 alpha를 비례적으로 조정하여 타임스텝별 실효 학습률 유지.",
    },
    "network_train_unet_only": {
        "en": "Train only the DiT (U-Net). Text encoder weights are frozen. Recommended for most LoRA training.",
        "ko": "DiT(U-Net)만 학습. 텍스트 인코더 가중치는 동결. 대부분의 LoRA 학습에 권장.",
    },
    "network_weights": {
        "en": "Path to a pre-trained adapter checkpoint to warm-start from. Leave empty for plain LoRA training.",
        "ko": "워밍업으로 사용할 사전 학습 어댑터 체크포인트 경로. 일반 LoRA 학습 시에는 비워두세요.",
    },
    "dim_from_weights": {
        "en": "Read network_dim from the warm-start checkpoint instead of the form value. Set together with network_weights so rank matches the warm-start LoRA.",
        "ko": "network_dim을 폼 값 대신 워밍업 체크포인트에서 읽기. network_weights와 함께 설정하여 랭크를 워밍업 LoRA와 일치시킵니다.",
    },
    # Training
    "learning_rate": {
        "en": "Base learning rate for the optimizer. Typical: 1e-5 to 1e-4.",
        "ko": "옵티마이저 기본 학습률. 일반적: 1e-5 ~ 1e-4.",
    },
    "max_train_epochs": {
        "en": "Total training epochs. One epoch = one full pass through the dataset.",
        "ko": "총 학습 에폭 수. 1 에폭 = 데이터셋 전체를 1회 순회.",
    },
    "save_every_n_epochs": {
        "en": "Save a checkpoint every N epochs. Set equal to max_train_epochs to save only the final model.",
        "ko": "N 에폭마다 체크포인트 저장. max_train_epochs와 같게 설정하면 최종 모델만 저장.",
    },
    "checkpointing_epochs": {
        "en": "Save resumable training state every N epochs. State files are large; use a larger interval than save_every_n_epochs.",
        "ko": "N 에폭마다 학습 재개 상태 저장. 상태 파일이 크므로 save_every_n_epochs보다 큰 간격 권장.",
    },
    "gradient_accumulation_steps": {
        "en": "Accumulate gradients over N steps before updating. Effective batch size = batch_size × accumulation_steps.",
        "ko": "N 스텝 동안 그레이디언트 누적 후 업데이트. 실효 배치 크기 = batch_size × accumulation_steps.",
    },
    "use_shuffled_caption_variants": {
        "en": "Consume preprocessed caption-shuffle variants from the text-encoder cache. When the cache holds multiple variants, a random one is drawn per sample. Falls back silently to single-variant if no variants were preprocessed.",
        "ko": "전처리된 캡션 셔플 변형을 텍스트 인코더 캐시에서 사용. 캐시에 여러 변형이 있으면 샘플당 무작위 선택. 변형이 전처리되지 않았다면 단일 캡션으로 자동 대체.",
    },
    "caption_dropout_rate": {
        "en": "Probability per sample of dropping the caption (replaced with empty text embedding). Pushes the LoRA toward an unconditional bias — useful for style training where you want the look to apply regardless of prompt. Typical: 0.0–0.05 for character/concept LoRAs, 0.1–0.25 for style LoRAs (그림체 학습). Too high can blur prompt-driven diversity (pose/composition).",
        "ko": '샘플별로 캡션을 비울(빈 텍스트 임베딩으로 대체) 확률. LoRA를 무조건부(unconditional) 방향으로 학습시켜, 프롬프트와 무관하게 항상 적용되는 "스타일"을 학습할 때 유리. 일반적: 캐릭터/컨셉 LoRA는 0.0–0.05, 그림체 학습은 0.1–0.25. 너무 높이면 캡션이 담당하던 다양성(포즈/구도)까지 함께 약해짐.',
    },
    "optimizer_type": {
        "en": "Optimizer algorithm. AdamW8bit: memory-efficient 8-bit Adam. Others: AdamW, Lion, Prodigy, etc.",
        "ko": "옵티마이저 알고리즘. AdamW8bit: 메모리 효율적 8비트 Adam. 기타: AdamW, Lion, Prodigy 등.",
    },
    "lr_scheduler": {
        "en": "Learning rate schedule. constant: fixed LR. Others: cosine, cosine_with_restarts, polynomial.",
        "ko": "학습률 스케줄. constant: 고정 LR. 기타: cosine, cosine_with_restarts, polynomial.",
    },
    "timestep_sampling": {
        "en": "How denoising timesteps are sampled during training. sigmoid: biased toward middle timesteps (recommended for flow matching).",
        "ko": "학습 중 디노이징 타임스텝 샘플링 방법. sigmoid: 중간 타임스텝 편향 (flow matching 권장).",
    },
    "discrete_flow_shift": {
        "en": "Flow-matching shift parameter controlling the noise schedule distribution. Default: 1.0.",
        "ko": "노이즈 스케줄 분포를 제어하는 flow-matching 시프트 매개변수. 기본값: 1.0.",
    },
    # Performance
    "attn_mode": {
        "en": "Attention backend. flash4: FlashAttention-4 (Linux, fastest). flash: FlashAttention-2. flex: PyTorch flex attention (cross-platform).",
        "ko": "어텐션 백엔드. flash4: FlashAttention-4 (Linux, 최속). flash: FlashAttention-2. flex: PyTorch flex attention (크로스 플랫폼).",
    },
    "gradient_checkpointing": {
        "en": "Recompute activations during backward pass instead of storing them. Trades compute for VRAM. Essential for low-VRAM setups.",
        "ko": "역전파 시 활성값을 저장 대신 재계산. 연산으로 VRAM 절약. 저사양 필수.",
    },
    "unsloth_offload_checkpointing": {
        "en": "Offload gradient checkpoints to CPU RAM. Further VRAM reduction at cost of speed. Requires gradient_checkpointing=true.",
        "ko": "그레이디언트 체크포인트를 CPU RAM으로 오프로드. 속도 감소 대신 VRAM 추가 절약. gradient_checkpointing=true 필요.",
    },
    "blocks_to_swap": {
        "en": "Number of DiT blocks to swap between GPU and CPU. 0: all on GPU. Higher values = more CPU offloading for low VRAM.",
        "ko": "GPU와 CPU 간 스왑할 DiT 블록 수. 0: 전부 GPU. 높을수록 더 많이 CPU로 오프로드.",
    },
    "torch_compile": {
        "en": "Enable torch.compile for the forward pass. Faster training after initial compilation. Best with static_token_count=true.",
        "ko": "torch.compile 활성화. 초기 컴파일 후 학습 속도 향상. static_token_count=true와 함께 사용 권장.",
    },
    "compile_mode": {
        "en": "'blocks': compile each DiT block individually (default). 'full': compile entire model as one graph for cross-block memory optimization. Full mode is incompatible with gradient checkpointing and block swap.",
        "ko": "'blocks': 각 DiT 블록을 개별 컴파일 (기본값). 'full': 전체 모델을 하나의 그래프로 컴파일하여 블록 간 메모리 최적화. full 모드는 gradient checkpointing 및 block swap과 호환 불가.",
    },
    "trim_crossattn_kv": {
        "en": "Remove zero-padding from cross-attention KV for efficiency. Flash4 applies LSE correction to maintain correct softmax.",
        "ko": "효율을 위해 크로스 어텐션 KV에서 제로 패딩 제거. Flash4는 정확한 softmax를 위해 LSE 보정 적용.",
    },
    "cache_llm_adapter_outputs": {
        "en": "Cache the LLM adapter layer outputs to disk. Avoids recomputing text encoder projections each epoch.",
        "ko": "LLM 어댑터 레이어 출력을 디스크에 캐싱. 매 에폭 텍스트 인코더 투영 재계산 회피.",
    },
    "masked_loss": {
        "en": "Apply loss only to non-masked regions (e.g., exclude text bubbles). Requires mask files in post_image_dataset/masks/ (or legacy masks/merged/) — run `make mask` to generate them.",
        "ko": "마스크되지 않은 영역에만 손실 적용 (예: 말풍선 제외). post_image_dataset/masks/ (또는 레거시 masks/merged/) 에 마스크 파일 필요 — `make mask` 로 생성.",
    },
    "mixed_precision": {
        "en": "Mixed precision mode. bf16: recommended for modern GPUs. fp16: for older GPUs without bf16 support.",
        "ko": "혼합 정밀도 모드. bf16: 최신 GPU 권장. fp16: bf16 미지원 구형 GPU용.",
    },
    "static_token_count": {
        "en": "Fixed 4096 token count for all batches. Gives torch.compile a single static shape — no recompilation across aspect ratios.",
        "ko": "모든 배치에 4096 토큰 고정. torch.compile에 단일 정적 셰이프 제공 — 화면비별 재컴파일 없음.",
    },
    "vae_chunk_size": {
        "en": "VAE decoding chunk size. Larger = faster but more VRAM. 64 is a good balance.",
        "ko": "VAE 디코딩 청크 크기. 클수록 빠르지만 VRAM 더 사용. 64가 적절.",
    },
    "vae_disable_cache": {
        "en": "Disable VAE's internal KV cache. Reduces VRAM during VAE encoding/decoding.",
        "ko": "VAE 내부 KV 캐시 비활성화. VAE 인코딩/디코딩 시 VRAM 감소.",
    },
    "cache_latents": {
        "en": "Cache VAE-encoded latents in memory. Avoids re-encoding images every epoch.",
        "ko": "VAE 인코딩된 레이턴트를 메모리에 캐싱. 매 에폭 이미지 재인코딩 회피.",
    },
    "cache_latents_to_disk": {
        "en": "Save cached latents to disk instead of RAM. Frees system memory at cost of disk I/O.",
        "ko": "캐시된 레이턴트를 RAM 대신 디스크에 저장. 디스크 I/O 대신 시스템 메모리 절약.",
    },
    "cache_text_encoder_outputs": {
        "en": "Cache text encoder outputs. Essential for lazy loading: encode → cache → free encoder → load DiT.",
        "ko": "텍스트 인코더 출력 캐싱. 지연 로딩 필수: 인코딩 → 캐시 → 인코더 해제 → DiT 로드.",
    },
    "cache_text_encoder_outputs_to_disk": {
        "en": "Save cached text encoder outputs to disk. Required for the lazy loading sequence to free VRAM before loading DiT.",
        "ko": "캐시된 텍스트 인코더 출력을 디스크에 저장. DiT 로드 전 VRAM 해제를 위한 지연 로딩 필수.",
    },
    "skip_cache_check": {
        "en": "Skip validation of cached files on startup. Faster startup when caches are known to be valid.",
        "ko": "시작 시 캐시 파일 검증 건너뛰기. 캐시가 유효함을 알 때 빠른 시작.",
    },
    "use_cmmd": {
        "en": "Use CMMD (PE-Core MMD²) as the validation signal. Off by default in the GUI — CMMD adds the PE encoder + a sampling pass per held-out item, which costs extra VRAM and time. Off → falls back to the cheaper per-σ FM-MSE val pass (uninformative on Anima but free).",
        "ko": "CMMD (PE-Core MMD²)를 검증 신호로 사용. GUI 기본값은 OFF — CMMD는 검증 항목마다 PE 인코더와 샘플링 패스를 추가해 VRAM과 시간 비용이 큼. OFF면 더 저렴한 σ별 FM-MSE 검증으로 대체 (Anima에서 유의미한 신호는 아니지만 무료).",
    },
    "use_valid": {
        "en": "Hold out a small validation slice from the training set (16 images by default — pinned via base.toml's validation_split_num). When off, the whole pool is used for training and no validation pass runs — skips the FM-MSE / CMMD eval and lets very small datasets train without losing samples. Writes/strips a {validation_split_num = 0, validation_split = 0.0} override on the variant's [[datasets]] block; base.toml is not touched.",
        "ko": "학습 셋에서 검증용 일부(기본 16장 — base.toml의 validation_split_num에서 지정)를 분리. 끄면 전체 풀이 학습에 쓰이고 검증 패스(FM-MSE / CMMD)는 실행되지 않음 — 데이터셋이 매우 작아 한 장도 빼기 싫을 때 유용. 변환 파일의 [[datasets]] 블록에 validation_split_num = 0, validation_split = 0.0 오버라이드를 쓰거나 제거함; base.toml은 건드리지 않음.",
    },
    "validation_split_num": {
        "en": "How many images to hold out for validation when 'use_valid' is on. Base.toml ships 16 — large enough for stable paired CMMD, small enough to barely dent the train pool. Bigger values give a quieter CMMD curve at the cost of fewer training samples and a slower val pass. Ignored when 'use_valid' is off (the zero-override wins). Writes/strips a {validation_split_num = N} override on the variant's [[datasets]] block; matching base.toml's value strips the override so base wins.",
        "ko": "'use_valid'가 켜져 있을 때 검증용으로 분리할 이미지 수. base.toml 기본값은 16 — paired CMMD가 안정적으로 측정될 만큼 크면서 학습 풀에 거의 영향을 주지 않는 크기. 값을 늘리면 CMMD 곡선이 더 매끄러워지지만 학습 샘플이 줄고 검증 패스도 느려짐. 'use_valid'가 꺼져 있으면 무시됨(0 오버라이드가 우선). 변환 파일의 [[datasets]] 블록에 {validation_split_num = N} 오버라이드를 쓰거나 제거함; base.toml과 같은 값이면 오버라이드가 제거되어 base가 우선됨.",
    },
    # Paths
    "pretrained_model_name_or_path": {
        "en": "Path to the base DiT model weights (.safetensors).",
        "ko": "기본 DiT 모델 가중치 경로 (.safetensors).",
    },
    "qwen3": {
        "en": "Path to the Qwen3 text encoder weights for text-to-image conditioning.",
        "ko": "텍스트-투-이미지 컨디셔닝용 Qwen3 텍스트 인코더 가중치 경로.",
    },
    "vae": {
        "en": "Path to the VAE model for image encoding/decoding.",
        "ko": "이미지 인코딩/디코딩용 VAE 모델 경로.",
    },
    "output_dir": {
        "en": "Directory for saving trained LoRA checkpoints.",
        "ko": "학습된 LoRA 체크포인트 저장 디렉토리.",
    },
    "output_name": {
        "en": "Base filename for saved checkpoints (epoch number is appended automatically).",
        "ko": "저장되는 체크포인트의 기본 파일명 (에폭 번호 자동 추가).",
    },
    "save_model_as": {
        "en": "Checkpoint format. safetensors: recommended (fast, safe).",
        "ko": "체크포인트 형식. safetensors: 권장 (빠르고 안전).",
    },
    "source_image_dir": {
        "en": (
            "Where raw images and .txt captions live. The Preprocess button feeds "
            "this to resize_images.py (writes resized PNGs) and "
            "cache_text_embeddings.py (caches captions). Override per preset/method "
            "if you keep multiple datasets side by side."
        ),
        "ko": (
            "원본 이미지와 .txt 캡션이 있는 디렉토리. 전처리 버튼이 이 경로를 "
            "resize_images.py(리사이즈된 PNG 저장)와 cache_text_embeddings.py"
            "(캡션 캐시)에 전달합니다. 여러 데이터셋을 병행할 때 프리셋/메소드별로 "
            "오버라이드하세요."
        ),
    },
    "resized_image_dir": {
        "en": (
            "Where preprocess writes VAE-aligned PNGs. Also resolved into the dataset "
            "subset's image_dir at training time (via {resized_image_dir} template "
            "in base.toml), so editing this propagates to both preprocess and training."
        ),
        "ko": (
            "전처리가 VAE에 맞춰 리사이즈한 PNG를 저장하는 디렉토리. 학습 시 "
            "데이터셋 서브셋의 image_dir로도 사용됩니다(base.toml의 "
            "{resized_image_dir} 템플릿 치환). 이 값을 바꾸면 전처리와 학습 양쪽에 "
            "반영됩니다."
        ),
    },
    "lora_cache_dir": {
        "en": (
            "Where preprocess writes VAE latent (.npz) and text-encoder "
            "(_anima_te.safetensors) caches. Also resolved into the dataset subset's "
            "cache_dir at training time."
        ),
        "ko": (
            "전처리가 VAE 잠재 변수(.npz)와 텍스트 인코더 출력"
            "(_anima_te.safetensors) 캐시를 저장하는 디렉토리. 학습 시 데이터셋 "
            "서브셋의 cache_dir로도 사용됩니다."
        ),
    },
    "path_pattern": {
        "en": (
            "fnmatch glob applied to each image's path relative to its subset's "
            "image_dir. Lets one variant TOML train on a slice of the full "
            "dataset without re-running preprocessing. `|` separates "
            "alternatives — OR-combine multiple patterns. Examples: `*` keeps "
            "everything; `char_a/*` keeps only files under the char_a/ "
            "subfolder; `char_a/*|char_b/*` keeps either folder; "
            "`*portrait*` keeps anything whose path contains 'portrait'. "
            "Validation enumeration and the image-count threshold honour the "
            "filtered pool."
        ),
        "ko": (
            "각 이미지의 image_dir 기준 상대 경로에 적용되는 fnmatch 글롭. "
            "전처리를 다시 돌리지 않고도 하나의 variant TOML로 전체 데이터셋의 "
            "일부만 학습할 수 있습니다. `|`로 여러 패턴을 OR 결합할 수 있습니다. "
            "예: `*`는 전체 사용, `char_a/*`는 char_a/ 하위만, "
            "`char_a/*|char_b/*`는 두 폴더 모두, `*portrait*`는 경로에 "
            "'portrait'이 포함된 파일만. 검증 데이터 열거와 이미지 개수 "
            "임계값도 필터링된 풀을 기준으로 동작합니다."
        ),
    },
}


def field_help(key: str) -> str | None:
    """Return the help string for *key* in the current language, or None."""
    entry = FIELD_HELP.get(key)
    if entry is None:
        return None
    lang = current_language()
    return entry.get(lang) or entry.get("en")


# ── Preprocessing tab field help ───────────────────────────────
# Distinct from the training-time `caption_dropout_rate` entry above:
# these knobs are consumed by preprocess/cache_text_embeddings.py at
# cache-build time, not by the dataloader.
PREPROCESS_FIELD_HELP: dict[str, dict[str, str]] = {
    "caption_shuffle_variants": {
        "en": (
            "Number of caption variants generated per image during text-encoder "
            "caching. v0 is the pristine original caption; v1..v(N-1) are smart-"
            "shuffled (the @artist prefix and 'On the …' / 'In the …' section "
            "anchors are preserved). The dataloader picks v0 with 20% probability "
            "and v1..v(N-1) uniformly otherwise — but only when "
            "use_shuffled_caption_variants=true in your method config. Set to 0 "
            "to cache a single pristine caption only."
        ),
        "ko": (
            "텍스트 인코더 캐싱 시 이미지당 생성되는 캡션 변형 수. v0은 원본 "
            "캡션 그대로이고, v1..v(N-1)은 스마트 셔플됩니다 (@artist 접두사와 "
            "'On the …' / 'In the …' 섹션 앵커는 보존). "
            "데이터로더는 v0을 20% 확률로 선택하고 나머지는 v1..v(N-1) 균등 "
            "분포로 선택합니다 — 단, 메소드 설정에 "
            "use_shuffled_caption_variants=true 일 때만 적용됩니다. "
            "0으로 설정하면 원본 캡션 하나만 캐싱합니다."
        ),
    },
    "caption_tag_dropout_rate": {
        "en": (
            "Per-tag dropout probability applied only to v1..v(N-1) shuffle "
            "variants. Tags up to and including the first @artist marker are "
            "never dropped — the artist tag is structurally important and the "
            "rating/character/series prefix is order-sensitive. Ignored when "
            "shuffle variants ≤ 0. Typical: 0.05–0.15. Higher values teach the "
            "LoRA to generalize across missing tags but can dilute the signal."
        ),
        "ko": (
            "v1..v(N-1) 셔플 변형에만 적용되는 태그별 드롭아웃 확률입니다. "
            "첫 번째 @artist 마커까지의 태그는 절대 드롭되지 않습니다 — "
            "작가 태그는 구조적으로 중요하고, 등급/캐릭터/작품 접두사는 "
            "순서가 중요합니다. 셔플 변형 수가 0 이하이면 무시됩니다. "
            "일반적: 0.05–0.15. 값이 높을수록 누락된 태그에 강건해지지만 "
            "학습 신호가 희석될 수 있습니다."
        ),
    },
    "run_sam_mask": {
        "en": (
            "Whether to run SAM3 bubble segmentation during 'Run masking'. "
            "Uncheck to skip SAM entirely — useful when SAM produces false "
            "positives on your dataset or when you only want MIT's text "
            "detection. If both SAM and MIT are unchecked, the Run button "
            "is blocked."
        ),
        "ko": (
            "'마스킹 실행' 시 SAM3 말풍선 분할을 실행할지 여부. 체크 해제 "
            "하면 SAM을 완전히 건너뜁니다 — 데이터셋에서 SAM이 오탐을 "
            "많이 내거나 MIT의 텍스트 검출만 원할 때 유용합니다. SAM과 "
            "MIT 둘 다 해제하면 실행 버튼이 차단됩니다."
        ),
    },
    "run_mit_mask": {
        "en": (
            "Whether to run MIT/ComicTextDetector text segmentation during "
            "'Run masking'. Uncheck to skip MIT — useful for non-manga "
            "datasets where its prior doesn't help, or to iterate faster on "
            "SAM-only configurations. If both SAM and MIT are unchecked, "
            "the Run button is blocked."
        ),
        "ko": (
            "'마스킹 실행' 시 MIT/ComicTextDetector 텍스트 분할을 "
            "실행할지 여부. 체크 해제하면 MIT를 건너뜁니다 — 만화가 아닌 "
            "데이터셋이나 SAM 전용 구성을 빠르게 시험할 때 유용합니다. "
            "SAM과 MIT 둘 다 해제하면 실행 버튼이 차단됩니다."
        ),
    },
    "sam_prompts": {
        "en": (
            "Text prompts SAM3 will look for in each image. One prompt per "
            "line. Defaults are tuned for manga-style speech / text bubbles. "
            "Add custom prompts if your dataset has additional regions you "
            "want masked out (e.g. 'sound effect', 'caption box'). Saved to "
            "configs/sam_mask.yaml — read directly by "
            "preprocess/generate_masks.py."
        ),
        "ko": (
            "SAM3이 각 이미지에서 찾을 텍스트 프롬프트. 한 줄에 하나씩. "
            "기본값은 만화 스타일 말풍선 / 텍스트 영역에 맞춰져 있습니다. "
            "추가로 마스킹하고 싶은 영역이 있으면 커스텀 프롬프트를 "
            "추가하세요 (예: 'sound effect', 'caption box'). "
            "configs/sam_mask.yaml에 저장되며 "
            "preprocess/generate_masks.py가 직접 읽습니다."
        ),
    },
    "sam_threshold": {
        "en": (
            "Minimum confidence required for a SAM3 detection to be kept. "
            "Range 0.0–1.0. Lower values produce more masks (with more false "
            "positives); higher values are stricter. Default 0.5. If SAM is "
            "missing real bubbles, lower this; if it's masking unrelated "
            "regions, raise it."
        ),
        "ko": (
            "SAM3 탐지를 유지하기 위한 최소 신뢰도. 범위 0.0–1.0. 값이 "
            "낮을수록 더 많은 마스크가 생성되지만 오탐도 늘어나며, 값이 "
            "높을수록 엄격해집니다. 기본값 0.5. SAM이 실제 말풍선을 "
            "놓치면 값을 낮추고, 관련 없는 영역을 마스킹하면 높이세요."
        ),
    },
    "sam_dilate": {
        "en": (
            "Pixels of binary dilation applied to each SAM mask after "
            "thresholding. Larger values blur mask edges outward — useful "
            "when the underlying segmentation undershoots the actual text "
            "bubble border. Default 5. Set to 0 to disable dilation."
        ),
        "ko": (
            "임계값 처리 후 각 SAM 마스크에 적용되는 이진 팽창 픽셀 수. "
            "값이 클수록 마스크 가장자리가 바깥으로 번집니다 — 실제 "
            "말풍선 경계보다 분할 결과가 작을 때 유용합니다. 기본값 5. "
            "0으로 비활성화."
        ),
    },
    "mit_text_threshold": {
        "en": (
            "Confidence threshold for the MIT/ComicTextDetector text "
            "segmenter. Range 0.0–1.0. Independent of SAM threshold — MIT "
            "uses a different model trained specifically on manga text "
            "regions. Default 0.8. Lower if MIT is missing text inside "
            "panels; raise if it's catching non-text artifacts."
        ),
        "ko": (
            "MIT/ComicTextDetector 텍스트 분할기의 신뢰도 임계값. 범위 "
            "0.0–1.0. SAM 임계값과는 별개입니다 — MIT는 만화 텍스트 "
            "영역에 특화된 다른 모델을 사용합니다. 기본값 0.8. 패널 "
            "내부 텍스트를 놓치면 낮추고, 텍스트가 아닌 부분을 잡으면 "
            "높이세요."
        ),
    },
    "mit_dilate": {
        "en": (
            "Pixels of binary dilation applied to each MIT mask. Same role "
            "as SAM dilate but tuned independently — MIT typically segments "
            "tight bounding regions around individual glyphs, so a moderate "
            "dilate value joins them into per-bubble blobs. Default 5."
        ),
        "ko": (
            "각 MIT 마스크에 적용되는 이진 팽창 픽셀 수. SAM 팽창과 같은 "
            "역할이지만 독립적으로 조정 — MIT는 일반적으로 개별 글리프 "
            "주변에 타이트한 경계를 분할하므로, 적당한 팽창 값으로 "
            "말풍선 단위 블롭으로 합쳐줍니다. 기본값 5."
        ),
    },
}


def preprocess_field_help(key: str) -> str | None:
    """Per-field help for the Preprocessing tab. Falls back to FIELD_HELP."""
    entry = PREPROCESS_FIELD_HELP.get(key)
    if entry is None:
        return field_help(key)
    lang = current_language()
    return entry.get(lang) or entry.get("en")


def preprocess_guide() -> str:
    return _guide("preprocess")


# ── Method guide dispatch ─────────────────────────────────────
# Methods that can't be baked into a plain DiT via scripts/merge_to_dit.py
# (router is layer-local / hook-only / not a weight delta) — render the
# "not mergeable" callout above their guide.
_NOT_MERGEABLE = frozenset({"postfix", "hydralora", "reft", "fera"})
_KNOWN_METHODS = frozenset({"lora", "tlora", "postfix", "hydralora", "reft", "fera"})


def method_guide(method: str) -> str | None:
    """Right-panel default HTML for *method*, or None if no guide is registered."""
    if method not in _KNOWN_METHODS:
        return None
    parts = [_guide("_apply_note")]
    if method in _NOT_MERGEABLE:
        parts.append(_guide("_not_mergeable"))
    parts.append(_guide(method))
    return "".join(parts)
