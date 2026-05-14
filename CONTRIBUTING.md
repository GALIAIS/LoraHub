# Contributing to LoraHub

Thanks for considering a contribution. LoraHub is in pre-alpha and the surface area is moving fast — small, well-scoped PRs are easier to review and ship.

## Quick checklist

Before opening a PR:

- `ruff check lorahub tests` passes
- `mypy lorahub` passes
- `python -m pytest -q` passes
- New behavior has tests
- Commits follow Conventional Commits (`feat:`, `fix:`, `refactor:`, etc.)
- Branch is named `feat/<short>` or `fix/<short>` and rebased on `main`

## Local setup

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[dev]"
python -m pytest -q
```

For backend integration work you also need an existing kohya-ss/sd-scripts checkout. Point at it with the `LORAHUB_KOHYA_SD_SCRIPTS` environment variable or with `backend.sd_scripts_path` in the recipe under test.

## Commit style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>

<body explaining why, not what>
```

Types: `feat`, `fix`, `refactor`, `style`, `docs`, `test`, `chore`. Subject is lowercase imperative under 50 chars; body wraps at 72.

## Scope guidelines

- Keep PRs focused. A bug fix and an unrelated cleanup belong in two PRs.
- Don't add features, abstractions, or extra config beyond what the task requires.
- Don't add docstrings or comments for self-evident code. Comment the *why*, not the *what*.

## Project layout

See [README.md](README.md#project-layout) for the directory map. The hot zones for v0.1 contributions are:

- `lorahub/core/backends/kohya/` — most kohya parameter coverage gaps live here
- `recipes/` — well-tuned recipes for common scenarios are very welcome
- `tests/` — every backend translation deserves a unit test

## Reporting bugs

Use the bug report template under [Issues](https://github.com/GALIAIS/LoraHub/issues/new/choose). A minimal recipe + the exact `lorahub` command + the relevant lines from the events.jsonl will get you fixed faster than anything else.

## Code of conduct

By participating you agree to abide by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
