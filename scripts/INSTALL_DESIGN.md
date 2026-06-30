# Install scripts — design contract

`scripts/install.sh` (POSIX / Linux / WSL) and `scripts/install.bat`
(Windows cmd) implement the same 6-step bootstrap. Keeping them in
sync is a load-bearing invariant: every change to one **must land
in the other in the same commit**, or users on the other platform
silently regress.

This document is the canonical step list. Anything not here is
platform-specific scaffolding (path syntax, error-output style).

## The 6 steps

Both scripts execute in this exact order. Each step is idempotent —
re-running the installer skips already-completed steps and only
fixes the ones that drifted.

| # | Step                | Output path        | Skip condition                                   |
| - | ------------------- | ------------------ | ------------------------------------------------ |
| 1 | Download `uv`       | `.lorahub/uv/`     | binary already present and reports its version   |
| 2 | Install Python 3.12 | `.lorahub/python/` | `uv python find 3.12` returns a path under it    |
| 3 | Create venv         | `.venv/`           | venv exists and `python --version` says 3.12     |
| 4 | Install lorahub     | `.venv/`           | always refresh editable package + dependencies   |
| 5 | Download Node.js    | `.node/`           | portable Node is at least `20.19.0`              |
| 6 | `npm ci`            | `web/node_modules/`| `package-lock.json` is in sync (npm `ci` would   |
|   |                     |                    | be a no-op)                                      |

All output paths are **relative to the repo root** and stay inside
the project tree, so `rm -rf <repo>` is the complete uninstall.

## Mirror knobs

Both scripts honour the same env vars:

| Variable                   | Step | Purpose                                          |
| -------------------------- | ---- | ------------------------------------------------ |
| `LORAHUB_GH_PROXY`         | 1    | GitHub proxy prefix for `uv` release tarball     |
| `UV_PYTHON_INSTALL_MIRROR` | 2    | `python-build-standalone` mirror (uv reads this) |
| `UV_INDEX_URL`             | 4    | PyPI index for `uv pip install`                  |
| `LORAHUB_NODE_MIRROR`      | 5    | Node binary tarball base (default nodejs.org)    |
| `NPM_CONFIG_REGISTRY`      | 6    | npm registry (npm reads this natively)           |
| `LORAHUB_TORCH_INDEX_URL`  | backend bootstrap | PyTorch wheel base URL selected by CN wrappers |

The China-friendly wrappers (`install-cn.sh` / `install-cn.bat`)
preset every variable in this table. Probe logic for picking the
fastest mirror lives in `scripts/install-cn-probe.ps1` and the
matching `case` in `install-cn.sh` — those are the **other** pair
that needs to stay in sync.

## When you change one script, change the other

A pre-flight checklist:

- [ ] Step number, output path, and skip condition match the table above.
- [ ] If you added a knob, both scripts honour it and this table lists it.
- [ ] Error messages are functionally equivalent (the user-visible text
  may differ for cmd vs bash idioms).
- [ ] The change has been smoke-tested on both platforms (or marked
  in the PR with "Linux only — Windows tracked in #N").

If the change is large enough that double-implementing it feels like
busywork, that's the signal to extract a helper. The current state
(two parallel scripts) is intentional: it's been chosen over a
common Python helper because the bootstrap cannot assume Python
exists yet — the whole point of step 2 is to put it there. A helper
could only cover steps 4-6, which would shave maybe 80 lines per
script and cost a `uv run` boundary in error reporting.

## Related scripts (not covered here)

* `scripts/install-cn.sh` / `install-cn.bat` / `install-cn-probe.ps1`
  — China mirror probe + wrapper. Picks the fastest mirror from a
  pool then forwards to the regular installer with env vars set.
* `scripts/run.sh` / `run.bat` — start the API server. No install
  logic; reuses the venv from step 3.
* `scripts/remote_setup.sh` / `remote_serve.sh` — VPS bootstrap
  variants. Currently free-standing; if the matrix grows further
  they should consume this design doc too.
