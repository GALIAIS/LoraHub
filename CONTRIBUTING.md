# 贡献指南

## 提 issue 之前

- 翻一下 [Issues](https://github.com/GALIAIS/LoraHub/issues)，确认问题没人提过。
- bug report 请附最小可复现的 config、命令行，以及 `runs/<job>/events.jsonl` 中的相关行。
- 涉及 `external/anima_lora/` 的 bug 请直接去上游 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 提；本仓库只接 LoraHub 集成层（`lorahub/core/backends/anima_lora/`）的修复。

## 提 PR 之前

- 在最新的 `main` 上拉一条短分支：`feat/<short>` 或 `fix/<short>`。
- 跑测试：`pytest tests/ -q`。
- 改了前端再跑一遍构建：`cd web && npm run build`。
- 跑 lint / 类型检查：`ruff check lorahub tests` 和 `mypy lorahub`。
- 较大的改动（schema 字段、新后端、跨模块重构）先在 issue 里同步设计。

## 本地开发

```powershell
git clone https://github.com/GALIAIS/LoraHub
cd LoraHub
pip install -e ".[api,dev]"
pytest tests/ -q
```

要做后端集成开发，需要可用的 kohya / diffusion-pipe checkout，可以在 config 里写 `backend.sdScriptsPath`，也可以用环境变量：

```powershell
set LORAHUB_KOHYA_SD_SCRIPTS=C:\path\to\sd-scripts
set LORAHUB_DIFFUSION_PIPE=C:\path\to\diffusion-pipe
```

Linux/macOS 用 `export`。也可把 `.env.example` 复制为 `.env` 后编辑。

## Commit 规范

格式：`<type>(<scope>): <subject>`，subject 用英文小写祈使句，不加句号，50 字以内。

| 类型       | 用途                                  |
| ---------- | ------------------------------------- |
| `feat`     | 新功能                                |
| `fix`      | bug 修复                              |
| `refactor` | 重构（不改外部行为）                  |
| `style`    | 代码格式 / 空白                       |
| `docs`     | 仅文档                                |
| `test`     | 仅测试                                |
| `chore`    | 工具链 / 构建 / 依赖                  |

scope 可选，写模块名（`auth`、`scheduler`、`anima_lora` 之类）。body 解释「为什么」，每行 72 字以内。引用 issue 用 `Closes #123`；破坏性变更在 footer 加 `BREAKING CHANGE:`。

示例：

```
feat(scheduler): resume sweep state across restarts
fix(anima_lora): preserve OrthoLoRA orthogonality after merge
```

## 第三方 vendored 代码

- `external/anima_lora/` 是 [sorryhyun/anima_lora](https://github.com/sorryhyun/anima_lora) 的固定 snapshot，按上游 license 随附。
- **不接受针对 `external/` 的直接 PR**——bug 修复请去上游。LoraHub 这边只接集成层（`lorahub/core/backends/anima_lora/`）的改动。
- 升级 vendored 版本走单独的 PR：commit message 写清上游 commit hash，PR 描述列出 user-visible 变化。

## 行为准则

参与项目即视为接受 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。
