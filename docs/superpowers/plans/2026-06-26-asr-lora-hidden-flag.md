# ASR-LoRA Hidden Flag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an off-by-default hidden `per_sample_timestep_mask` network arg for the modified `anima_lora` T-LoRA path.

**Architecture:** Reuse the existing T-LoRA mask surface and only change mask shape when the new flag is true. Standard LoRA and old T-LoRA stay unchanged because the default remains the existing batch-mean `(1, R)` mask.

**Tech Stack:** Python, PyTorch tensors, existing `LoRANetworkCfg`, existing `LoRANetwork.set_timestep_mask()`.

---

### Task 1: Config and Kwarg Surface

**Files:**
- Modify: `external/anima_lora/networks/__init__.py`
- Modify: `external/anima_lora/networks/lora_anima/config.py`

- [ ] **Step 1: Add hidden network arg**

Add `"per_sample_timestep_mask"` immediately after `"use_timestep_mask"` in `SHARED_KWARG_FLAGS`.

- [ ] **Step 2: Add config field**

Add this field after `use_timestep_mask` in `LoRANetworkCfg`:

```python
per_sample_timestep_mask: bool = False
```

- [ ] **Step 3: Parse config field**

In `LoRANetworkCfg.from_kwargs()`, parse:

```python
per_sample_timestep_mask = _as_bool(kwargs.get("per_sample_timestep_mask"))
```

and pass it into the constructor next to `use_timestep_mask`.

### Task 2: Per-Sample Mask in T-LoRA

**Files:**
- Modify: `external/anima_lora/networks/lora_anima/network.py`

- [ ] **Step 1: Keep old path unchanged**

At the top of `set_timestep_mask()`, branch:

```python
if not self.cfg.per_sample_timestep_mask:
    # existing batch-mean mask path
```

- [ ] **Step 2: Add new path**

For the new path, allocate a shared mask with shape `(B, 1, R)` and compute rank from each sample timestep:

```python
t = timesteps.float().reshape(-1)
frac = ((max_timestep - t) / max_timestep).clamp(min=0.0, max=1.0)
r = frac.pow(self.cfg.alpha_rank_scale) * (max_rank - self.cfg.min_rank) + self.cfg.min_rank
r = r.clamp(max=float(max_rank))
mask.copy_((self._timestep_mask_arange.unsqueeze(0) < r.unsqueeze(1)).to(mask.dtype).unsqueeze(1))
```

### Task 3: Test

**Files:**
- Test: `tests/test_lora_research_rank_mask.py`

- [ ] **Step 1: Keep research utility test**

Use the existing `per_sample_rank_mask()` test as the shape/value oracle.

- [ ] **Step 2: Run checks**

Run:

```powershell
.venv\Scripts\python.exe -m py_compile external/anima_lora/networks/lora_anima/config.py external/anima_lora/networks/lora_anima/network.py
.venv\Scripts\python.exe -m pytest tests/test_lora_research_rank_mask.py -q
```

Expected here: py_compile passes; pytest may skip locally if torch is absent.
