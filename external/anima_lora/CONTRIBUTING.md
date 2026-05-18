# Contributing

Thanks for considering a contribution. This repo welcomes targeted fixes and new adapter methods. Read the right tier below before opening a PR — the bar is different for each.

## Before you start

- Open or comment on a [GitHub issue](https://github.com/sorryhyun/anima_lora/issues) describing the change. For anything bigger than Tier 1, please get a thumbs-up on scope before writing code — saves a round-trip on direction.
- Read [`CLAUDE.md`](CLAUDE.md) end-to-end. It is the single source of truth for the config flow, training invariants, and architecture. Most "is this how things work?" questions are answered there.
- Use `uv` for everything (`uv sync`, `uv run python …`). Don't add `pip` instructions to docs or commit `requirements.txt` files.
- Run the linters before pushing:
  ```bash
  ruff check . --fix && ruff format .
  ```

## Priority areas

Five areas where outside contributions would have the biggest impact right now. Each item below carries a tier annotation that maps to the requirements in the rest of this document. Open a draft PR or issue early on anything bigger than Tier 1 — happy to scope and review.

### 1. EasyControl adapters

Per-block cond LoRA on self-attn + FFN with a logit-bias gate. DiT frozen; trains a handful of cond LoRA blocks plus a scalar gate. The architecture is naturally contribution-friendly: each control type is one independent adapter, no method changes required. See [`docs/experimental/easycontrol.md`](docs/experimental/easycontrol.md). The wall right now is the **adapter zoo** around it.

- **Trained adapters** — canny, depth, pose, lineart, scribble, segmentation, … each one a self-contained PR with model card, training config, and samples. Hosted under a HuggingFace collection (planned: `anima-easycontrol`). *[Tier 1.5 — bench numbers and side-by-side samples carry the PR; no new method code]*
- **Per-task dataset spec** — one doc per control type covering pair format, recommended size (~2k pairs), where to source signal images. Currently undocumented. *[Tier 1]*
- **Toy datasets** — 200-pair CC-licensed bundles per control type so a contributor can validate the pipeline before committing to a full dataset. *[Tier 1]*
- **One-command training aliases** — `make easycontrol-canny`, `make easycontrol-depth`, … as per-task preset configs in `configs/methods/easycontrol/`. *[Tier 1]*
- **Control-fidelity eval harness** — held-out ~100-pair sets per control type that re-extract the signal from generation (canny→canny, depth→depth, …) and report a fidelity metric vs the input. Lets adapter PRs be reviewed on numbers rather than vibes. The current `bench/easycontrol/` directory has equivalence + smoke scripts only; the harness slot is empty. *[Tier 1.5]*

### 2. Turbo LoRA (Decoupled DMD distillation)

Distill 28-step Anima @ CFG=4 into a 4–8 step generator using **co-LoRA** (LoRA for both the student and the fake score model on the same frozen DiT). The deployment story is that `turbo_anima_lora.safetensors` stacks on top of any existing concept LoRA at inference, the same way LCM-LoRA composes with style LoRAs. See [`docs/proposal/turbo_anima_dmd_lora.md`](docs/proposal/turbo_anima_dmd_lora.md) — Decoupled DMD reference: Liu et al., arXiv:2511.22677.

Status: proposal only — no code, no checkpoints, no bench. The proposal is fully scoped (file-level plan, phased validation, risk register) and is waiting on an implementer.

What's missing — this is one Tier 2 PR by definition (new method + paper + `bench/turbo/` + docs/methods entry + `make exp-turbo` / `make exp-test-turbo`), but it splits cleanly along phase boundaries:

- **Phase 0: single-prompt overfit (~1 day).** Implement `networks/methods/turbo_dmd.py` (two LoRA networks, attachment toggle), `scripts/distill_turbo.py` (CA + DM gradient assembly, two optimizer states, the renoise primitive), `configs/methods/turbo.toml`, `make exp-turbo`. Prove the loop converges on one prompt at batch 1, 2k iterations. *[Tier 2 — drop a `bench/turbo/results/<ts>-phase0/` with teacher@28 vs student@4 side-by-side on a fixed seed]*
- **Phase 1: 100-prompt sweep (~3 days).** Image Reward + HPS v2.1 + per-aspect breakdown (1024², 832×1248, 1248×832). Pass = student IR ≥ 80% of teacher, no aspect below 60%. *[Tier 2 continuation]*
- **Phase 2: full HPS bench (~1 week).** 1k COCO-prompt sample, all 4 schedule configs from the paper's Table 1 as an ablation, replicates the paper's Decoupled-Hybrid claim on Anima. *[Tier 1.5 once Phase 1 has landed]*
- **Phase 3: composition test (~2 days).** (turbo only) vs (concept LoRA @ 28) vs (turbo + concept @ 4) on three existing concept checkpoints. Validates the deployment story. *[Tier 1.5]*

If Phase 1 fails after one rank bump, the proposal explicitly says kill it — don't grind. The phase gates are there to bound the contributor's downside.

### 3. DCW calibration

DCW v4 (`make dcw` → `fusion_head.safetensors`) ships. The wall is calibration coverage: each released LoRA needs its own fusion head, and several v4 controller paths are stubbed. See [`docs/methods/dcw.md`](docs/methods/dcw.md) "Limitations / open questions".

- **σ̂² channel re-train (3-seed rerun).** Prototype fails Gate B on the variance head. Default is to ship with `--dcw_v4_disable_shrinkage` until this clears. Run `make dcw` across 3 independent seeds, retrain `train_fusion_head.py` with the combined pool, re-evaluate Gate B. If it still fails, document the failure and harden the disable flag as permanent. *[Tier 1.5 — bench script exists, the contribution is the seed sweep + write-up]*
- **Tiled inference path.** v4 controller currently no-ops under tiled VAE/DiT. The single-tile `c_pool` / `g_obs` is ill-defined at tile boundaries. Two paths: (a) compute one global `c_pool` / `g_obs` before tiling and broadcast it across tiles, (b) keep the no-op and document. Either is a valid PR; (a) is preferred. *[Tier 1.5]*
- **CFG drift.** v4 is calibrated at CFG=4 only — the production setting. CFGs other than 4 fall back to scalar. A CFG=1 / CFG=7 calibration pool + a `--dcw_v4_cfg_select` switch would close the gap. The CFG=1 case in particular intersects with paper-direction sign-flips (`project_dcw_cfg_aspect_signflip`). *[Tier 2-shaped — new calibration pool counts as a bench artifact]*
- **Cached-Spectrum `x0_pred` ablation.** v4 should still help under Spectrum because the correction is bias-agnostic, but the Chebyshev forecaster biases `x0_pred` and the row hasn't been measured. One bench run, one ablation table. *[Tier 1.5]*
- **Per-released-LoRA fusion heads.** Each major LoRA release should ship its calibrated head as a sibling artifact (`<lora_name>.fusion_head.safetensors`). The current `make dcw` is incremental — the contribution is running it on each released checkpoint and publishing the head. *[Tier 1 — operational, no code]*
- **`scripts/dcw/` documentation pass.** `measure_bias.py`, `train_fusion_head.py`, `haar.py`, `trajectory.py`, `collect_fei_sidecar.py` are under-commented. Reviewing each as a contributor would, and improving the docstrings / `--help` strings, is welcome. *[Tier 1]*

### 4. Filling the bench gaps

The `bench/<method>/` convention from Tier 2 below requires every method bench to ship a `README.md` (what it measures, run command, output layout, baseline run, interpretation). Several existing subdirs predate that requirement and are missing it:

| Subdir | Status | What it has | What's needed |
|---|---|---|---|
| `bench/dcw/` | **No README** | `measure_bias.py`, `train_fusion_head.py`, `covariance_ceiling.py`, `k_supervision_sweep.py`, `stability_predictor_check.py`, `sweep_buckets.py`, `transfer_hypothesis_check.py`, `plot_seed_band.py` — large `results/` corpus | One README that maps each script to a finding and links to a canonical results dir |
| `bench/easycontrol/` | **No README** | `step0_equivalence.py`, `step1p5_lse_equivalence.py`, `two_stream_smoke.py` | README + the control-fidelity harness slot from (1) above |
| `bench/fera/` | **No README** | `probe_fei.py`, `probe_fei_dataset.py`, `probe_fei_3band_dataset.py`, `probe_closed_loop.py`, `expressivity_analysis.py`, `refactor_lowdim_forward.py` | README explaining the probe ladder + linking to the 2-band decision in `project_fera_probe_2band_decision` |
| `bench/hydralora/` | **No README** | `bench.py`, `analyze_drift.py`, `prompts.example.txt`, plus `hydralora_proposal.md` + `progress0421.md` | README that promotes the proposal/progress notes into a runnable-bench entry |
| `bench/spectrum/` | Has README | analytical drift simulator + image bench | None — use this one as the shape template |

Each missing README is a self-contained Tier 1 PR. Use `bench/spectrum/README.md` as the model: headline, what each script does, a copy-pasteable run command, the headline number(s) and what "good" looks like, links to representative `results/<timestamp>/` runs, and an "Observed on Anima" section.

A second-order bench-gap contribution worth calling out:

- **Envelope conformance.** Older bench scripts predate `bench/_common.py` and don't drop a `result.json` via `make_run_dir` + `write_result`. Auditing each script and converting the holdouts (so cross-run indexing actually works) is a clean Tier 1 PR per script.
- **`bench/turbo/` doesn't exist yet** — it lands as part of the Turbo LoRA contribution in (2) above.

### 5. Translations & localization

Translatable content lives in four places, each with its own contribution shape but all reviewed as Tier 1 (no bench, no test — `make gui` walkthrough screenshots in the PR description are the proof). Missing entries in every surface below **fall back to English**, so it's fine to ship an incomplete translation and grow it over time. Currently shipped: `en` (canonical), `ko` (mostly complete), `cn` (machine-translated stub, unproofread).

**(a) GUI strings — `gui/i18n/<code>.py`.** One module per language, each exporting `STRINGS: dict[str, str]`. `gui/i18n/__init__.py` assembles these into `TRANSLATIONS` and `t(key, **kwargs)` resolves keys against the current language. To **add a new language**, drop in `gui/i18n/<code>.py` mirroring `en.py`'s key set, register it in `TRANSLATIONS`, and add a friendly label to `LANG_NAMES` in `gui/app.py` (e.g. `"ja": "日本語"`). Every key you do include must use the same `{placeholder}` names as the English source — `t()` calls `.format(**kwargs)` and will raise at runtime on a typo. *[Tier 1]*

**(b) Per-field tooltips — `gui/explanations/__init__.py`.** Two dict-of-dicts power the form-field help: `FIELD_HELP` (config-form tooltips, ~50 keys) and `PREPROCESS_FIELD_HELP` (Preprocessing tab knobs, ~10 keys). Each entry is `{"en": "...", "ko": "..."}`. Add your language code as a sibling key in every entry you want translated. `field_help()` / `preprocess_field_help()` fall back to `"en"` for missing language keys. These are the strings users see when they click a form-row label, so translation quality matters more than for transient buttons — keep technical terms (LoRA, MoE, σ-bucket, VAE) untranslated. *[Tier 1]*

**(c) Long-form method guides — `gui/explanations/guides/<name>.<lang>.html`.** Right-panel HTML blocks for method variants and the Preprocessing tab. Filename convention is `<name>.<lang>.html`; the loader (`_read_guide` in `gui/explanations/__init__.py`) auto-falls back to `.en.html` when the language version is absent. Names currently present: `lora`, `tlora`, `hydralora`, `fera`, `reft`, `postfix`, `preprocess`, plus the shared snippets `_apply_note` and `_not_mergeable`. To translate, drop in `<name>.<code>.html` files alongside the English ones — no code change required. Preserve any `<a href="…">`, `<code>`, and color-coded `<span>` markup; the GUI's QTextBrowser renders these. *[Tier 1]*

**(d) Docs and structure images — `docs/`.**
- `docs/guidelines/가이드북.md` is the end-to-end onboarding doc and only exists in Korean. An English translation (or any other language) would significantly widen the audience. The `guidebook_tooltip` string in `gui/i18n/en.py` currently points users at the Korean file — once a translation lands, wire the Guidebook button (in `gui/app.py`) to pick the right file based on `current_language()`.
- `docs/structure_images_korean/` holds Korean-labeled versions of the architecture diagrams under `docs/structure_images/` (e.g. `animakor.png` ↔ `anima.png`). English/other-language equivalents are welcome under the natural sibling tree (`docs/structure_images/` is the English baseline; `docs/structure_images_<lang>/` for translations). Mention which markdown files reference the diagram so the reviewer can update the embed paths.
- Method docs under `docs/methods/`, `docs/experimental/`, `docs/proposal/`, and `docs/optimizations/` are **English-only by convention** — translations are welcome as `<name>.<code>.md` siblings, but nothing reads them at runtime yet. If you contribute one, also propose how it should surface (e.g. a language switcher in the README's docs table, or wiring it into a GUI "Open method doc" button). Don't translate `CLAUDE.md` — that file is consumed by Claude Code and is single-source-of-truth for project conventions.

**Parity check (covers all per-language surfaces):**
```bash
python -c "
from gui.i18n import TRANSLATIONS as T
from gui.explanations import FIELD_HELP, PREPROCESS_FIELD_HELP
en_keys = set(T['en'])
for lang in T:
    if lang == 'en': continue
    missing = sorted(en_keys - set(T[lang]))
    print(f'{lang} i18n missing  ({len(missing)}):', missing[:5], '…' if len(missing) > 5 else '')
for name, d in (('FIELD_HELP', FIELD_HELP), ('PREPROCESS_FIELD_HELP', PREPROCESS_FIELD_HELP)):
    en_entries = {k for k, v in d.items() if 'en' in v}
    for lang in ('ko', 'cn'):
        missing = sorted(k for k in en_entries if lang not in d[k])
        print(f'{lang} {name:24s} missing ({len(missing)}):', missing[:5], '…' if len(missing) > 5 else '')
"
```

If a translated string is too long for its widget, mention it in the PR — the fix is usually a layout tweak in the relevant `gui/tabs/*.py`, not a shorter translation.

## Tier 1 — bug fixes, typos, UI, arg/CLI tweaks

Lightweight contributions. Examples: fixing a regex in a LoRA target list, a typo in a docstring, a confused error message, a GUI label, a missing CLI flag, a `tasks.py` argument-forwarding bug.

**Requirements:**
- Existing tests pass:
  ```bash
  make test-unit
  ```
- The change is minimal and scoped. No drive-by refactors, no new abstractions, no "while I'm here" reformatting in unrelated files.
- For GUI changes, actually launch the GUI (`make gui`) and exercise the affected tab before claiming the PR is done. Type-checking is not a substitute for clicking the button.
- For training-path changes, smoke-test one short run end-to-end (`PRESET=low_vram make lora` truncated to a few steps is fine) and paste the tail of the log into the PR description.

That's it. Open the PR.

## Tier 1.5 — efficiency improvement or algorithm revision

A change that touches an existing method's compute path, scheduling, or numerics — without introducing a new method. Examples: a faster kernel for an existing attention path, replacing an FP32 reduction with a lower-precision one, revising T-LoRA's mask schedule, tweaking HydraLoRA's router temperature handling, changing the LSE correction in `attention_dispatch.py`, swapping the optimizer step order for memory.

These sit between Tier 1 and Tier 2: no new paper or new docs page is required, but **the burden of proof is empirical** — you are claiming the existing method runs faster, uses less memory, or produces equivalent-or-better outputs under a revised algorithm. That claim has to be measurable.

**Requirements:**

1. **Bench script.** A runnable script that quantifies the change. Two acceptable shapes:
   - **Add to an existing `bench/<method>/`** if the change is scoped to one method (e.g. a HydraLoRA router tweak goes under `bench/hydralora/`). Append a new script and a new section to that bench's README.
   - **Add a small `bench/<topic>/`** for cross-cutting changes (e.g. an attention dispatch optimization belongs in something like `bench/attention/`).

   The script must report the headline number(s) it claims to move — wall-clock, peak VRAM, loss-at-N-steps, drift, whatever the change targets — for **both before and after**. A single-number claim ("20% faster") with no reproducible script does not clear the bar. If the script loads the DiT, use `bench/_anima.py` (`add_common_args` + `build_anima`) — same rationale as Tier 2 §2 below: every DiT-loading bench needs to expose `--compile` and load the adapter in the right order, and the helper enforces both.

2. **New or extended tests.** At least one test that locks in the invariant the change is supposed to preserve. Examples:
   - For a kernel rewrite: a numerical-equivalence test against the previous path within a stated tolerance.
   - For a schedule revision: a test that the new schedule reduces to the old one under a documented config flag, so the change can be A/B'd.
   - For a memory optimization: an assertion on peak allocator usage on a small fixture, if feasible.

   Add the test to `tests/`, following the patterns in `test_network_registry.py` and `test_lora_custom_autograd.py`. If exact equivalence is impossible (e.g. a deliberately different algorithm), state the tolerance and what would constitute a regression.

3. **Documentation update.** Update the relevant `docs/methods/<name>.md`, `docs/optimizations/<name>.md`, or section of `CLAUDE.md` to reflect the new behavior. No new top-level doc unless the change introduces a user-visible flag that warrants one.

4. **Result in the PR description.** Paste the bench output (before/after) and the test results into the PR description. Link to the bench script that produced them. Reviewers should be able to reproduce the claim with one command.

5. **Backwards-compat statement.** If the change alters numerics (loss curves shift, output images change at fixed seed), say so explicitly. If it does not, say that and explain why — bit-equivalent refactors and behavior-changing optimizations get reviewed differently.

A paper citation is welcome but not required. If the revision is paper-derived, cite the paper as you would in Tier 2; if it's a hand-rolled improvement, the bench results stand on their own.

## Tier 2 — new LoRA / adapter method

A new entry in `networks/lora_modules/` or `networks/methods/`, or a new variant block in `configs/methods/lora.toml` / a new `configs/methods/<name>.toml`.

**Requirements:**

1. **Paper reference.** New methods exist because someone published a result that justifies the complexity. The PR description must cite the paper (title, authors, venue, arXiv id) and the upstream code if any. Method docs follow the same format as the existing ones — see `docs/methods/reft.md` (shipped) and `docs/experimental/easycontrol.md` (experimental) for the shape. Stable methods land in `docs/methods/<name>.md`; unstable / unmerged-into-shipped methods land in `docs/experimental/<name>.md`.

   Hand-rolled methods without prior art are not categorically rejected, but the bar is higher: in the absence of a paper, the bench results have to carry the argument alone, and reviewers will be skeptical. If you are confident, propose the method in an issue first.

2. **Dedicated bench subdirectory.** Create `bench/<method_name>/` with the same shape as the existing ones (`bench/spectrum/`, `bench/dcw/`, `bench/easycontrol/`, `bench/hydralora/`):

   ```
   bench/<method_name>/
   ├── README.md              # what the bench measures, how to run, how to read output
   ├── proposal.md            # (optional) design framing — why this method, what it should beat
   ├── plan.md                # (optional) integration plan if the bench is an early diagnostic
   ├── <bench_script>.py      # a runnable script, not a notebook
   └── results/               # gitignored except for the timestamped run you cite in the PR
   ```

   Wire the script's output through `bench/_common.py` so it produces a standard `result.json` envelope (script path, git SHA, env, args, metrics, artifacts) under `results/<YYYYMMDD-HHMM>[-<label>]/`. The two helpers are:

   ```python
   from bench._common import make_run_dir, write_result

   out_dir = make_run_dir("<method_name>", label=args.label)
   # ... write CSVs / PNGs / etc. into out_dir ...
   write_result(out_dir, script=__file__, args=args,
                metrics={...}, artifacts=[...], device=device)
   ```

   **Benches that load the DiT must use `bench/_anima.py`.** It owns the model-side boilerplate — argparse surface, DiT + adapter loading in the correct order, bucketed sample discovery from `post_image_dataset/lora/`. `add_common_args(parser)` injects `--label`, `--seed`, `--device`, `--dtype`, `--attn_mode`, `--gradient_checkpointing`, `--cpu_offload_checkpointing`, `--compile`, `--compile_mode`. `build_anima(args, adapter=..., train_mode=...)` does the load in the right order — in particular, **`compile_blocks` runs after `apply_to` + `load_weights`** so adapter monkey-patches are part of the compiled graph. Open-coding this is a footgun (skipping `--compile` entirely, or compiling in the wrong order so the adapter is bypassed). Use the helper:

   ```python
   from bench._anima import add_common_args, build_anima, discover_bucketed_samples

   p = argparse.ArgumentParser()
   p.add_argument("--dit", required=True)
   p.add_argument("--adapter", default=None)
   add_common_args(p)
   args = p.parse_args()

   bundle = build_anima(args, adapter=args.adapter, train_mode=False)
   anima, network = bundle.anima, bundle.network
   ```

   Benches that don't load the DiT (analytical simulators, post-hoc result-aggregators) don't import `bench/_anima.py` — both modules are opt-in.

   The bench README must include:
   - **What it measures** — the headline number(s) and what "good" looks like.
   - **Run command** — copy-pasteable, defaults reasonable, runs on a single 12–16 GB GPU in under 30 minutes.
   - **Output layout** — what files land alongside `result.json` under `results/<YYYYMMDD-HHMM>[-<label>]/`.
   - **Interpretation** — what the numbers mean, including what would falsify the method.
   - **Baseline run** — at least one results directory checked in (or linked from a release artifact if large), with the exact CLI used to produce it.

   `bench/dcw/README.md` is a good template — it documents the measurement, has an "Observed on Anima" section with a dated baseline, and a "Next actions" section. Aim for that.

3. **Documentation.** A method doc at `docs/methods/<name>.md` covering the algorithm, config knobs, training/inference flow, and known failure modes. Cross-link from the README's "Experimental features" table.

4. **Tests.** At least a smoke test that constructs the network and runs one forward pass on CPU/CUDA. Existing tests in `tests/` show the shape (`test_network_registry.py`, `test_loss_registry.py`, `test_smoke.py`).

5. **Make/`tasks.py` entry points.** A new method needs `make <name>` and matching `python tasks.py <name>` invocations, plus a `test-<name>` target that runs `inference.py` against a checkpoint produced by the method. Follow the patterns in the `Makefile`.

6. **Mergeability statement.** If the method produces weights that fold into the base DiT (LoRA family), confirm that `make merge` works and ship a merge-equivalence check in the bench. If it does not (ReFT / Hydra moe / postfix / prefix / IP-Adapter / EasyControl), say so explicitly in the doc and update `scripts/merge_to_dit.py`'s refusal list.

7. **Empirical result.** The PR must show the method works on Anima specifically. Cite a bench run from `bench/<method_name>/results/<timestamp>/` and link to a small set of side-by-side images (3–6 seeds is fine) demonstrating the claimed effect. "It compiles and trains without crashing" is not a result — both `LoRA + this` and `LoRA alone` need to be in the comparison.

## Tier 3 — new base-model support

**Currently not accepted.**

This repo is Anima-specific by design. Adding a second base model is a multi-week project that touches the trainer forward path, every adapter monkey-patch, the cache filename convention, and every `configs/methods/*.toml` LoRA target list. The blocker is `train.py::get_noise_pred_and_target` and the per-adapter Anima coupling, not the DiT class itself. See [`docs/multi_model_support.md`](docs/multi_model_support.md) for the full terrain map and effort estimate.

What is in scope:
- **Improving `docs/multi_model_support.md`** — sharper coupling map, more accurate effort estimates, concrete protocol sketches, a worked example of what a `ModelFamily` port would look like for a specific candidate model. Pure-doc PRs of this kind are welcome.
- **Decoupling work that has standalone value on Anima** — e.g. parameterizing cache suffixes, lifting the LoRA target regex into a `lora_target_spec()`, moving strategy base classes up. If a refactor makes the Anima code cleaner *and* incidentally reduces the multi-model blocker, propose it as its own PR with the Anima-side justification leading.

What is not in scope:
- A second `library/models/<family>/` namespace populated for a real second model.
- A new `forward_for_loss` slot on a hypothetical `ModelFamily` protocol that nothing else uses yet.
- Caches, configs, or test fixtures for a second model.

If you want to fork the repo to support a different base model, that's fine and encouraged — but the upstream stays Anima-only until a maintainer decides otherwise.

## PR checklist

Copy this into your PR description and tick what applies:

- [ ] Tier identified (1 / 1.5 / 2 / 3-eligible doc work).
- [ ] `make test-unit` passes locally.
- [ ] `ruff check` and `ruff format` clean.
- [ ] (Tier 1.5) Bench script added or extended; before/after numbers in the PR description.
- [ ] (Tier 1.5) New or extended test locking in the invariant the change preserves.
- [ ] (Tier 1.5) Backwards-compat statement: numerics-equivalent or behavior-changing.
- [ ] (Tier 2) Bench subdirectory present with README, runnable script, and a timestamped baseline run.
- [ ] (Tier 2) Paper citation in the PR description and method doc.
- [ ] (Tier 2) `docs/methods/<name>.md` added and cross-linked from `README.md`.
- [ ] (Tier 2) `make <name>` and `make test-<name>` work.
- [ ] (Tier 2) Merge story documented (folds into DiT? if not, why not?).
- [ ] No commented-out code, no `print(...)` debug leftovers, no unrelated formatting churn.

## License

By contributing you agree your changes are licensed under the same license as this repo (see `LICENSE`).
