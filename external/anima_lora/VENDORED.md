# Vendored: anima_lora

This directory is a snapshot of the upstream `anima_lora` training
engine, vendored into LoraHub so the workbench can drive it without
asking users to clone a separate tree at install time.

The lorahub-side wrapper lives at `lorahub/core/backends/anima_lora/`.
The vendored source itself stays as upstream-faithful as possible —
patches go in named files under [`patches/`](patches/) (see below) so
they can be re-applied on top of a future sync.

## Upstream

| Field                | Value                                                  |
| -------------------- | ------------------------------------------------------ |
| Source repository    | https://github.com/sorryhyun/anima_lora                |
| License              | Apache 2.0 (`LICENSE` in this directory)               |
| Project version      | `0.1.0` (`pyproject.toml`)                             |
| Last sync date       | 2026-05-24                                             |
| Frozen commit        | partial — base import 2026-05-22, plus selected fixes from upstream `5360c5f` (2026-05-23). The 2026-05-24 native-flatten refactor (`75e121d`) is intentionally NOT applied yet. |

The "frozen commit" line is unknown because the vendored copy was
imported by file copy rather than as a git submodule, so we don't
have an upstream SHA to pin against. Re-syncs from upstream should
populate this field — see [Re-syncing from upstream](#re-syncing-from-upstream)
below.

## Why vendor instead of submodule

* **Single-clone install.** New users running `scripts/install.sh`
  get the trainer ready without `git submodule update --init`.
* **Pinned dependency surface.** Upstream's `pyproject.toml` is
  free to drift between commits; vendoring lets us roll the bump
  through CI before exposing it to users.
* **Patch points.** A handful of small modifications (registered
  via the lorahub plugin registry — see
  [Local modifications](#local-modifications)) sit on top of upstream;
  a submodule would force users to re-apply them every clone.

## Local modifications

Changes vendored on top of upstream. Each entry should describe the
*intent* so a future re-sync can reapply or discard intelligently.

* `networks/lora_modules/` — supplemented with the 10 PEFT
  algorithm implementations LoraHub registered (DoRA, IA3, LoKr, LoHA,
  DyLoRA, Full, Diag-OFT, BOFT, GLoRA, VeRA). Filenames mirror the
  algorithm enum on `AnimaLoraMethodLoraConfig.algorithm`.
* `networks/__init__.py` — accepts LyCORIS/kohya-compatible
  `network_args` selectors (`algo=locon|loha|lokr|ia3|dylora|full|diag-oft|boft|glora|tlora`)
  and maps them onto the native anima_lora `NetworkSpec` registry.
  `factor=` is normalized to `lokr_factor=` for LoKr compatibility.
* `gui/`, `output/`, `post_image_dataset/` — runtime scratch the
  trainer writes during a run. Excluded from git via the repo-root
  `.gitignore`.
* `pyproject.toml` — keeps `aiohttp` constrained to stable 3.x so the
  global `prerelease = "allow"` setting used for torch nightlies does
  not pull `aiohttp 4.0.0a1` source builds on CPython 3.13.

### Selected upstream fixes (2026-05-24 cherry-pick)

Applied piecewise from upstream commits past the 2026-05-22 base import:

* **Soft-tokens contrastive OOM fix** (upstream `5360c5f` /
  `6e8cb01`): replaced `networks/methods/soft_tokens.py` wholesale
  with the 5360c5f version that introduces `_pending_gradcache` +
  `after_backward` deferred replay, so the contrastive negatives
  can compose with `blocks_to_swap > 0` without OOM. Wired the
  generic post-backward hook surface alongside it:
  `library/training/method_adapter.py` gains an
  `after_backward(ctx)` no-op default; `train.py` adds
  `run_after_backward(ctx)`; `library/training/loop.py` calls
  `trainer.run_after_backward(state.train_ctx)` immediately after
  `accelerator.backward(loss)` and before gradient clipping.
* **IP-Adapter caption-dropout retune** (upstream `b277443`):
  `configs/methods/ip_adapter.toml` `caption_dropout_rate`
  bumped from `0.05` to `0.1` to reduce identity drift on
  strong-text prompts at converged-token training.

* **Sample-time CUDAGraphs step boundary** (LoraHub-side patch,
  no upstream PR yet — 2026-05-24): inserted
  `torch.compiler.cudagraph_mark_step_begin()` at the top of each
  iteration in `library/anima/training.py::do_sample`. Without
  this, recipes that combine `--torch_compile` +
  `--compile_mode blocks` + `--compile_inductor_mode reduce-overhead`
  hit a RuntimeError on the first sample-image step ("accessing
  tensor output of CUDAGraphs that has been overwritten by a
  subsequent run") because the training loop's last compiled
  forward left the reused activation buffers populated. Marking
  the step boundary lets the runtime reclaim the slots safely.
  Wrapped in try/except so older torch builds without the helper
  silently skip (those builds default to no cudagraphs anyway).
* **June 2026 stability cherry-picks** (upstream `6c15855` /
  `8c2005c` / `fea3433` / `d47207e` / `6a89efe`, applied selectively
  on 2026-06-11): ported the useful backend-safe pieces without the
  GUI/runtime-harness migration. Sample/validation VAE decode now parks
  the DiT on CPU and moves decoded pixels back to CPU before training
  resumes; LoRA-family rank GEMMs avoid fp32 upcast from AdaLN fp32
  activations by computing in the model output dtype; stale
  torch.compile inductor caches are cleared when the compile shape
  signature changes. LoraHub also emits `use_cmmd = false` explicitly
  and train.py prints `validation loss=...` so the wrapper can surface
  validation events.
* **Checkpoint layout + SVD-Down backports** (reviewed against upstream
  `71c7c8c`, applied selectively on 2026-06-22): epoch/step checkpoints
  and their matching `*-state` directories now write under a per-run
  `<output_name>/` directory, while final and rolling auto-resume files
  stay in the output root so LoraHub artifact discovery and resume logic
  remain compatible. The plain LoRA path also accepts
  `network_args = ["down_init=weight_svd"]`, a conservative backport of
  upstream's SVD-Down initialization from `2674b59`; LoraHub exposes it
  as `backend.animaLora.lora.downInit` for `algorithm="lora"` and
  `algorithm="tlora"` only. Text-embedding preprocessing now encodes
  missing caption sidecars as `""`, matching the training dataset loader
  and upstream's missing-caption cache behavior.

### Intentionally deferred upstream changes

These were reviewed and held back because they would break the
LoraHub control surface (schema / compiler / policies / tests) and
require a coordinated wrapper migration:

* **Native-flatten compile refactor** (upstream `75e121d`,
  2026-05-24): removes `set_static_token_count(pad=True)`,
  `compile_core`, `--compile_mode full`, `static_token_count`,
  `static_pad`. Replaces them with `compile_blocks()` +
  `_native_flatten`, and migrates the bucket table to two
  token-count families (4032 + 4200) — invalidating the
  4096-token-only `CONSTANT_TOKEN_BUCKETS` we still ship and
  every disk cache built against it. LoraHub `compiler.py` /
  `policies.py` / `schema.py` still reference the removed
  field names (`static_token_count`, `compile_mode='full'`)
  and validate against them; lifting these requires a parallel
  schema migration and is left for a follow-up.
* **Distill-mod / synth migration to native shapes** (upstream
  `88456f8`): depends on the same native-flatten refactor.
* **Daemon / GUI / installer changes** (upstream `9baaf76`,
  `b277443`): not vendored — LoraHub drives anima_lora through
  its own job runner / API / installer, so the upstream
  daemon / PySide6 / install.ps1 surface is intentionally
  out-of-tree (see "Excluded paths" / "Why vendor instead of
  submodule" above).
* **Upstream GUI / daemon queue changes** (through upstream `71c7c8c`,
  2026-06-22): upstream's latest commits are primarily PySide caption
  editor UI, GUI preload, and daemon queue behavior. LoraHub does not
  use upstream's GUI or daemon queue, so these are intentionally not
  copied.
* **Turbo distillation refactor + SVD-Down turbo knobs** (upstream
  `2674b59`): deferred. Upstream moved turbo into
  `scripts/distill_turbo/config.py` / `distill.py` and threads
  `student_down_init` / `fake_down_init` through that new structure.
  LoraHub's vendored copy still uses the older single-file
  `scripts/distill_turbo.py`; partially adding the new knobs would
  create a schema surface the runner cannot execute safely.
* **Training knob pruning and uncond-sidecar relocation** (upstream
  `772dda7`, `380414f`, `eb20053`): deferred. LoraHub's schema,
  compiler, and resume/checkpoint policies still expose LR scheduler,
  checkpoint-rotation, CPU-offload, and the older uncond-sidecar layout;
  those need a coordinated wrapper migration before adoption.

If you add a non-trivial modification, drop the diff (or a brief
description) into `patches/NNNN_<slug>.md` so the next re-sync can
follow the trail.

## Excluded paths (root `.gitignore`)

These paths exist in the working tree but are *not* committed:

```
external/anima_lora/.codex
external/anima_lora/.venv/
external/anima_lora/__pycache__/
external/anima_lora/anima_lora.egg-info/
external/anima_lora/output/
external/anima_lora/post_image_dataset/
external/anima_lora/uv.lock          # tracked separately under bootstrap
```

## Re-syncing from upstream

When upstream pushes a fix or feature LoraHub wants:

1. Note the target upstream SHA (`git ls-remote https://github.com/sorryhyun/anima_lora`).
2. Diff this directory against an export at that SHA — for example:
   ```bash
   git clone --depth=1 --branch=<sha> https://github.com/sorryhyun/anima_lora /tmp/anima_lora_upstream
   diff -ruN /tmp/anima_lora_upstream external/anima_lora \
     --exclude=.venv --exclude=__pycache__ --exclude=anima_lora.egg-info \
     --exclude=output --exclude=post_image_dataset \
     > /tmp/anima_lora_local.patch
   ```
   The patch is the union of "upstream commits we don't have yet"
   *and* our local modifications.
3. Inspect the patch and split it into:
   * upstream commits to take in,
   * local modifications to keep,
   * obsolete local mods to drop.
4. Replace the directory contents with the new upstream tree, then
   re-apply the kept local mods.
5. Update the table at the top of this file (`Last sync date`,
   `Frozen commit`).
6. Run `pytest tests/test_anima_lora_*.py` to confirm the
   compiler / preprocess / preview wiring still passes.
7. Commit the result in a single atomic commit titled
   `chore(vendor): sync anima_lora to <sha>`.
