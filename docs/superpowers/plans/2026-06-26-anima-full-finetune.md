# Anima Full Finetune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full-model finetuning only to the modified `anima_lora` backend.

**Architecture:** Reuse the existing anima_lora training loop by adding a duck-typed `network_module` that exposes the same lifecycle methods as adapter networks but trains/saves the full DiT. LoraHub compiler selects that module when `backend.animaLora.method = "full_finetune"`; all other backends stay unchanged.

**Tech Stack:** Python, PyTorch, safetensors, Pydantic, React config form.

---

### Task 1: Backend Network Module

**Files:**
- Create: `external/anima_lora/networks/methods/full_finetune.py`

- [ ] Add a `FullFinetuneNetwork` class with `create_network`, `apply_to`, optimizer param methods, and `save_weights`.
- [ ] Save the full Anima DiT state dict with `library.anima.weights.save_anima_model`.

### Task 2: LoraHub Compiler and Schema

**Files:**
- Modify: `lorahub/core/config/backends/anima_lora.py`
- Modify: `lorahub/core/backends/anima_lora/compiler.py`
- Modify: `tests/test_anima_lora_compiler.py`

- [ ] Add `full_finetune` to the method enum.
- [ ] Emit `network_module = "networks.methods.full_finetune"` and skip LoRA `network_args` for that method.
- [ ] Add a compiler test asserting the generated TOML uses full finetune and omits LoRA flags.

### Task 3: Frontend Form

**Files:**
- Modify: `web/src/components/config-form/sections/backend-anima-lora-options.ts`
- Modify: `web/src/components/config-form/sections/backend-anima-lora.tsx`
- Modify: `web/src/components/config-form/sections/backend-anima-lora-methods.tsx`
- Modify: `web/src/components/config-form/types.ts`

- [ ] Add the method option.
- [ ] Hide LoRA rank/alpha and algorithm panel when `method === "full_finetune"`.
- [ ] Keep precision/cache/sampling controls visible.

### Task 4: Verification

- [ ] Run Python compile checks for touched backend files.
- [ ] Run the focused compiler test.
- [ ] Run frontend type/build check if available.
