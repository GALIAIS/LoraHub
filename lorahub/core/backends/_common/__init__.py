"""Shared helpers used by every concrete training backend.

The `kohya` and `diffusion_pipe` backends repeat a lot of plumbing: looking
up an existing checkout from a recipe field / env var / default location,
checking that a python interpreter exists, running `git clone` and uv steps
with the same progress + error-handling shape, etc. Those bits live here so
the per-backend modules only carry what is actually different (repo URL,
required-files list, requirements entry point, optional extras like
xformers / deepspeed).

Nothing in this package imports from a specific backend, so it is safe to
import from anywhere in the backends tree without circular dependencies.
"""
