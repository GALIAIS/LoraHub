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
| Last sync date       | 2026-05-22                                             |
| Frozen commit        | unknown — vendored snapshot, not a git submodule       |

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
* `gui/`, `output/`, `post_image_dataset/` — runtime scratch the
  trainer writes during a run. Excluded from git via the repo-root
  `.gitignore`.

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
