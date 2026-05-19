# Contributing to LoraHub / 贡献指南

LoraHub 处于 alpha 阶段，主要表面在持续演进。中文段在前，英文段紧随其后。

LoraHub is in alpha and the surface area is still moving. Chinese paragraphs come first, English follows.

## 提 issue 之前 / Before filing an issue

中文：

- 先看 [`docs/audit-2026-05.md`](docs/audit-2026-05.md)。已知问题、计划项和 backlog 都列在里面，避免重复提。
- 翻一下 [Issues](https://github.com/GALIAIS/LoraHub/issues)。
- bug report 请附最小可复现的 config、命令行和 `runs/<job>/events.jsonl` 中的相关行。
- 涉及 `external/anima_lora/` 的 bug 请直接去上游 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 提；本仓库只接 LoraHub 集成层（`lorahub/core/backends/anima_lora/`）的修复。

English:

- Read [`docs/audit-2026-05.md`](docs/audit-2026-05.md) first; known issues, plans, and backlog are tracked there.
- Search [Issues](https://github.com/GALIAIS/LoraHub/issues).
- For bug reports, include a minimal reproducing config, the exact command, and the relevant lines from `runs/<job>/events.jsonl`.
- Bugs in `external/anima_lora/` belong upstream at [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora). This repo only accepts fixes to the integration layer (`lorahub/core/backends/anima_lora/`).

## 提 PR 之前 / Before opening a PR

中文：

- 在最新的 `main` 上拉一条短分支：`feat/<short>` 或 `fix/<short>`。
- 跑测试：`pytest tests/ -q`。
- 改了前端的话再跑一遍构建：`cd web && npm run build`。
- 跑 lint / 类型检查：`ruff check lorahub tests` 和 `mypy lorahub`。
- Commits 用 [Conventional Commits](https://www.conventionalcommits.org/) 英文写。
- 较大的改动（schema 字段、新后端、跨模块重构）先开 issue 或 design doc。可以参考 [`docs/anima-lora-integration.md`](docs/anima-lora-integration.md) 的写法。

English:

- Branch off the latest `main`: `feat/<short>` or `fix/<short>`.
- Run `pytest tests/ -q`.
- If the PR touches the web app, also run `cd web && npm run build`.
- Run `ruff check lorahub tests` and `mypy lorahub`.
- Commits follow [Conventional Commits](https://www.conventionalcommits.org/) in English.
- For larger changes (schema fields, new backends, cross-module refactors), open an issue or design doc first. [`docs/anima-lora-integration.md`](docs/anima-lora-integration.md) is a worked example.

## 本地开发 / Local setup

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev]"
pytest tests/ -q
```

要做后端集成开发，需要有可用的 kohya / diffusion-pipe checkout，可以在 config 里写 `backend.sd_scripts_path`，也可以用环境变量：

For backend integration work, point LoraHub at an existing kohya / diffusion-pipe checkout via `backend.sd_scripts_path` or via env vars:

```powershell
$env:LORAHUB_KOHYA_SD_SCRIPTS = "C:\path\to\sd-scripts"
$env:LORAHUB_DIFFUSION_PIPE   = "C:\path\to\diffusion-pipe"
```

## Commit 规范 / Commit style

中文：

- 格式：`<type>(<scope>): <subject>`，subject 用英文小写祈使句，不加句号，50 字以内。
- type 限定：`feat`、`fix`、`refactor`、`style`、`docs`、`test`、`chore`。
- scope 可选，写模块名（`auth`、`scheduler`、`anima_lora` 之类）。
- body 解释“为什么”，每行 72 字以内。
- 引用 issue 用 `Closes #123`；破坏性变更用 `BREAKING CHANGE:` 段。

English:

- Format: `<type>(<scope>): <subject>`. Subject is lowercase imperative under 50 chars, no trailing period.
- Allowed types: `feat`, `fix`, `refactor`, `style`, `docs`, `test`, `chore`.
- Scope is optional; use module names like `auth`, `scheduler`, `anima_lora`.
- Body explains the why and wraps at 72 chars.
- Reference issues with `Closes #123`; mark breaking changes with a `BREAKING CHANGE:` footer.

示例 / examples:

```
feat(scheduler): resume sweep state across restarts
fix(anima_lora): preserve OrthoLoRA orthogonality after merge
```

## Vendored 第三方 / Vendored third-party code

中文：

- `external/anima_lora/` 是 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 的固定 snapshot，按上游 license 随附。
- **不接受针对 `external/` 的直接 PR**——bug 修复请去上游。LoraHub 这边只接集成层（`lorahub/core/backends/anima_lora/`）的改动。
- 升级 vendored 版本走单独的 PR：commit message 写清上游 commit hash，PR 描述列出 user-visible 变化。

English:

- `external/anima_lora/` is a pinned snapshot of [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora), bundled under its upstream license.
- **PRs that change files inside `external/` are not accepted.** Send bug fixes upstream. Changes in the integration layer (`lorahub/core/backends/anima_lora/`) are in scope.
- Vendored upgrades go through their own PR: include the upstream commit hash in the commit message and list user-visible changes in the PR description.

## 翻译贡献 / Translation contributions

中文：

- 欢迎翻译贡献。中文为权威语言，英文跟随中文。
- 如果中英两段有冲突，以中文为准；先在 PR 里提出来，再决定改哪一边。
- 不要只翻译不带对应中文修订（除非中文段已是最新且只缺英文）。

English:

- Translation PRs are welcome. Chinese is the authoritative language; English follows it.
- If the two diverge, the Chinese version wins; flag the divergence in the PR before changing either side.
- Don't translate without a matching Chinese update unless the Chinese version is already current and only the English side is missing.

## Code of conduct

参与项目即视为接受 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

By participating you agree to abide by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

