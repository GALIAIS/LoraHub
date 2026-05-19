# Image Studio — 设计文档

> Status: 已实现(B7 后路由拆分到 9 个子模块)
> Owner: LoraHub maintainer(单用户产品)
> Spec date: 2026-05-17

## 1. 这页解决什么

LoRA 质量瓶颈在 **数据集质量**。LoraHub 当前在三个相对独立的页面上处理数据
准备:Datasets(扫目录 + caption 预览)、wd14 / joytag 自动打标、Sample
Gallery(训练后的预览)。三者都没有给用户一个直接 **对图集本身做工** 的位
置 — 排重、评分、修裁切、批量改 200 张图的 caption,或让 VLM 补 wd14 看不
到的细节。

Image Studio 填这个空。它是一个图先行的工作台:用户打开一个训练用图目录,
直到目录"训练就绪"才离开。每个动作要么改磁盘上的数据集,要么写入元数据到
`runs/image_studio.sqlite`。

为什么这是 LoraHub 中最重要的 AI-touch 面:

- 对自然语言驱动的模型(Anima、FLUX、Wan),VLM caption 价值高于 wd14。
- 画质评分 + 构图描述把 300 张图的人工排查从数小时缩到数分钟。
- 多数用户不会切到外置图片编辑器。如果不在这一处把"编辑 + caption + AI 增
  强"凑齐,项目通常被搁置。

## 2. 不做什么

为避免 image-studio 长成 Photoshop 克隆,以下功能明确不做:

- 像素级绘制 / inpainting mask。(支持比例裁切 + 简单亮度 / 对比度 / 饱
  和度;更重的活让用户在自己惯用的图片编辑器里完成。)
- 超出 `bmp/gif/jpg/jpeg/png/webp` 的多格式转换。AVIF / HEIC 按需再加。
- 多用户并发编辑。LoraHub 是单用户。
- 云同步。文件留在用户放置的位置。
- 生成式 img2img / outpainting。明确不做 — 不跑推理服务,这是 ComfyUI
  的事。

## 3. 在 IA 中的位置

```
sidebar:
  数据面板 (dashboard)
  训练任务 (jobs)
  超参 sweep
  训练配置 (configs)
  数据集 (datasets)              ← 保留为薄薄的扫一眼即走入口
  图像工作台 (image-studio) NEW  ← 深度编辑器
  样图画廊 (gallery)
  设置 / 关于
```

**数据集** 页保持原状 — 粘路径、看扁平网格、跳到训练。它是索引。任何图片的
"打开工作台"按钮(或数据集卡片级"进入工作台"按钮)把目录交给 **图像工作
台**。image-studio 是编辑器。

Sample Gallery(`/gallery`)不受影响 — 那是训练输出,不是输入。

## 4. 信息架构

Image Studio 是单页(`/image-studio`),URL 携带路径状态:

```
/image-studio?path=<encoded-dir>&recursive=0&page=1
```

三栏布局:

```
┌────────────────────────────────────────────────────────────────────┐
│ Toolbar: 路径栏 + 递归 + 过滤 chip + AI 批操作                     │
├──────────┬──────────────────────────────────────────┬───────────────┤
│  filters │  image grid (虚拟化)                     │  inspector    │
│   panel  │                                          │  (选中图)     │
└──────────┴──────────────────────────────────────────┴───────────────┘
```

- **Toolbar**:沿用现有 `PathBar`、递归开关、计数 badge、"+ 选区"多选切
  换、"AI"下拉(批量 caption / 评分 / 重复建议 / trigger 词建议)。
- **过滤面板**(左 14rem):caption 覆盖 chip(有 / 缺)、AI 画质桶 chip
  (优 / 中 / 差 / 未评)、tag 多选(从已观测 tag)、纵横比 chip(横 /
  竖 / 方)、重复簇 chip、文件大小 + 分辨率 slider、日期范围。
- **图片网格**(中,虚拟化,支持 j/k 键移动、x 切换选中、e 编辑 caption、
  q 评分、del 软删)。
- **Inspector**(右 22rem):整图、EXIF / 尺寸,caption 编辑器(对比 AI
  建议)、带权 tag chip、AI 面板(对当前图运行 / 显示上次结果)、动作按
  钮(旋转 90、水平翻转、比例裁切、替换、软删)。

第二个 tab — `/image-studio?view=clusters` — 把重复簇排成横向 carousel,
每簇一个"留这个"radio。

## 5. State model

### 5.1 New SQLite database: `runs/image_studio.sqlite`

```sql
-- Per-image annotation. The image file itself stays on disk untouched
-- until the user clicks Save / Apply. All AI-enrich results land here.
CREATE TABLE image_annotations (
    image_path           TEXT PRIMARY KEY,    -- absolute, normalised
    sha256               TEXT NOT NULL,       -- for dedupe + stable id
    width                INTEGER,
    height               INTEGER,
    bytes                INTEGER,
    -- AI annotations
    ai_caption           TEXT,                -- last VLM caption
    ai_caption_provider  TEXT,                -- task route used
    ai_caption_at        TEXT,                -- ISO timestamp
    ai_quality_score     REAL,                -- 0.0 - 1.0
    ai_quality_label     TEXT,                -- 优 / 中 / 差
    ai_quality_reason    TEXT,                -- VLM rationale
    ai_quality_at        TEXT,
    ai_composition       TEXT,                -- VLM short description
    ai_composition_at    TEXT,
    ai_trigger_words     TEXT,                -- JSON array suggested
    ai_trigger_words_at  TEXT,
    -- User annotations
    user_quality_label   TEXT,                -- 优 / 中 / 差 / 排除
    user_notes           TEXT,
    soft_deleted         INTEGER NOT NULL DEFAULT 0,
    favorite             INTEGER NOT NULL DEFAULT 0,
    updated_at           TEXT NOT NULL
);

CREATE INDEX idx_image_sha256 ON image_annotations(sha256);
CREATE INDEX idx_image_quality ON image_annotations(ai_quality_label);
CREATE INDEX idx_image_user_quality ON image_annotations(user_quality_label);

-- Perceptual hash for near-duplicate detection (separate row per algo
-- so we can A/B different hashes without schema migration).
CREATE TABLE image_phash (
    image_path  TEXT NOT NULL,
    algo        TEXT NOT NULL,        -- "phash64" | "dhash64"
    hash        TEXT NOT NULL,        -- hex
    PRIMARY KEY (image_path, algo)
);
CREATE INDEX idx_phash_value ON image_phash(algo, hash);

-- Pending edits batched until the user clicks Apply. Lets us preview
-- "rotate + caption swap + trigger-word merge" before the file actually
-- changes on disk. One row per pending mutation.
CREATE TABLE image_pending_ops (
    id         TEXT PRIMARY KEY,
    image_path TEXT NOT NULL,
    op         TEXT NOT NULL,        -- "rotate" | "flip" | "crop" |
                                     -- "replace_caption" | "merge_caption"
                                     -- | "delete" | "favorite"
    payload    TEXT NOT NULL,        -- JSON
    created_at TEXT NOT NULL
);
```

The on-disk image is never mutated until the user applies pending ops.
Captions live in their canonical sidecar (`<image>.txt`) — Image Studio
never invents a second caption store; it edits the same file the
training job will read.

### 5.2 Reuse, don't duplicate

- **Caption sidecars** stay where they are. Same `/api/datasets/caption`
  read/write pair the captions tab already uses.
- **Thumbnails** reuse the existing `/api/datasets/thumb` cache at
  `runs/.thumbs/<sha256>.webp`.
- **AI calls** go through the ShiroManager-shaped `/api/ai/invoke` and
  the `tagging.assist` / `caption.rewrite` task routes already wired in
  Settings → AI 服务商.
- **Path allow-list** reuses `dataset_files._allowed_roots()` so the
  studio never reaches outside dataset roots / job workspaces.

## 6. Backend API

All endpoints live under `/api/image-studio/`. Existing
`/api/datasets/*` endpoints stay; the studio uses them for read +
caption I/O.

### 6.1 Listing & filters

```
GET /api/image-studio/list?path=<dir>&recursive=0&page=1&limit=48
    &filter.caption=any|with|missing
    &filter.quality=any|优|中|差|unscored|favourite
    &filter.aspect=any|landscape|portrait|square
    &filter.tag=<tag>&filter.tag=<tag>           multi-tag AND
    &filter.cluster=<cluster_id>
    &sort=name|mtime|size|quality|caption_len
->
{
  "path": "...",
  "total": 248,
  "page": 1,
  "limit": 48,
  "items": [{
    "path": "...", "relative_path": "...",
    "name": "...", "width": 1024, "height": 1536, "bytes": 412000,
    "mtime": 1716000000.0,
    "caption": "...", "caption_exists": true,
    "annotation": { ai_caption, ai_quality_label, ai_quality_score,
                    user_quality_label, soft_deleted, favorite, ... } | null,
    "thumb_url": "/api/datasets/thumb?path=...&size=256",
    "tags": ["1girl","blue_hair","..."]   -- parsed from caption
  }]
}
```

### 6.2 Per-image inspect

```
GET /api/image-studio/image?path=<file>
->
{ ...same item shape as above,
  "exif": { camera, lens, iso, exposure, ... },
  "phash": { phash64: "...", dhash64: "..." },
  "pending_ops": [...],
  "duplicates": [{ path, distance, sha256_match }]   -- nearest matches
}
```

### 6.3 Annotations CRUD (user-side)

```
PUT    /api/image-studio/annotations
       { path, user_quality_label?, user_notes?, favorite?, soft_deleted? }
DELETE /api/image-studio/annotations?path=<file>      -- clear AI+user fields
```

### 6.4 Pending ops

```
POST   /api/image-studio/ops
       { path, op, payload }              -- queue one
GET    /api/image-studio/ops?path=<file>  -- inspect queued
DELETE /api/image-studio/ops/{id}
POST   /api/image-studio/ops/apply        -- commit all pending for a path
       { path }
       Returns { applied: [...], errors: [...] }
```

Apply semantics per op:
- `rotate` / `flip`: re-encode in place, preserve EXIF orientation, bump mtime
- `crop`: same; payload `{ left, top, right, bottom }` in pixels
- `replace_caption` / `merge_caption`: write `<image>.txt`
- `delete`: move file + sidecar to `runs/_image_studio_trash/<date>/`
- `favorite`: flip `image_annotations.favorite` (no file mutation)

### 6.5 Similarity & duplicates

The studio offers **two layers** of similarity, with deliberate cost +
intent gradient:

| Layer | Signal | Cost | Catches |
|-------|--------|------|---------|
| L1 — Perceptual hash | `phash64` / `dhash64` over downscaled greyscale | local, ~ms/image | identical files, mild crops, recompresses |
| L2 — AI semantic | CLIP-style embedding via VLM **OR** pairwise VLM "are these the same scene?" | API call (token cost) | same outfit / pose / scene with different lighting, redrawn variants |

L1 always runs first to seed candidate buckets; L2 runs only on the
buckets you ask it to deepen. This keeps token spend bounded.

#### 6.5.1 L1 — perceptual hash scan

```
POST /api/image-studio/dedupe/scan
     { path, recursive?, algo: "phash64"|"dhash64", threshold?: int }
->   202 { session_id }                   -- background, progress streamed

GET  /api/image-studio/dedupe/clusters?path=<dir>&kind=phash
->   {
       clusters: [{
         id, kind: "phash",
         members: [{ path, distance_to_centroid, sha256_match }],
         suggested_keep: <path>,
         centroid_phash: "...",
       }]
     }
```

Default keep heuristic: highest resolution → largest file → lex-first
path. User can override per cluster, then commit.

#### 6.5.2 L2 — AI semantic similarity

Two modes; the user picks per scan:

**Mode A — embedding-based (recommended)**: a single VLM-embedding call
per image, results cached in `image_embeddings`. Once every image has
an embedding we cluster by cosine similarity. **Cheap, scales to 1000+
images, the right default**. Uses providers that expose
`/v1/embeddings` (OpenAI, voyage-ai-style, anything OpenAI-compatible).
Falls back to Mode B if the configured provider has no embedding model.

**Mode B — pairwise VLM judge** (fallback / refinement): for each
candidate pair from L1, send both images + the prompt "are these two
training images near-duplicates of the same subject/scene? Answer:
yes_strong | yes_weak | no". One round-trip per pair. Used to tighten
L1 false-positives or de-confuse a cluster the user is unsure about
("L1 says these are the same, AI confirm?").

```
POST /api/image-studio/similarity/scan
     {
       path, recursive?,
       mode: "embedding" | "pairwise",
       task: "similarity.score",      -- AI route id (new task)
       seed_clusters?: [<phash_cluster_id>],   -- B mode: refine these
       threshold?: float                       -- A mode: cosine threshold
                                               -- B mode: which verdicts to keep
     }
-> 202 { session_id }
```

The session reuses the tagging-session shape so the existing progress
bar component renders for free; per-cluster intermediate results are
streamed so the user can act on the first cluster while the rest is
still computing.

```
GET /api/image-studio/similarity/clusters?path=<dir>&kind=ai
-> {
     clusters: [{
       id, kind: "ai",
       confidence: 0.0-1.0,             -- mean cosine / mean verdict
       members: [{
         path,
         similarity_to_keep: 0.0-1.0,
         verdict: "yes_strong" | "yes_weak" | null,   -- B mode only
         ai_reason: "...",                            -- optional VLM rationale
       }],
       suggested_keep: <path>,
       suggested_delete: [<path>],     -- the rest, ranked worst-first
     }]
   }
```

`image_embeddings` table:

```sql
CREATE TABLE image_embeddings (
    image_path TEXT NOT NULL,
    model_id   TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    vector     BLOB NOT NULL,         -- float32 packed
    created_at TEXT NOT NULL,
    PRIMARY KEY (image_path, model_id)
);
```

The frontend defaults to **Mode A** with cosine threshold 0.92, surfaces
both clusters (L1 + L2) interleaved in the same Duplicates tab, lets
the user toggle "show L1 only / L2 only / both".

### 6.6 Batch delete on similarity clusters

The clusters tab is the primary "drop dataset weight" surface.
Single-cluster keep-radio is flow A; batch-select-many-clusters is
flow B.

UI shape (Duplicates tab):

```
┌────────────────────────────────────────────────────────────────┐
│ Layer chips: [phash] [AI 语义]            筛选: 阈值 [0.92 ▾]   │
│ 选择全部建议删除  共 38 张, 释放 ~412 MB     [批量删除]         │
├────────────────────────────────────────────────────────────────┤
│ Cluster #1  conf 0.97  保留 → A.png                             │
│   ☐ A.png  (keep, 1024x1536, sharp)         ← suggested keep    │
│   ☑ B.png  sim 0.99  AI: 同一姿势同光照                          │
│   ☑ C.png  sim 0.96  AI: 同一姿势, 服装 微变                     │
│   ☐ D.png  sim 0.91  AI: 不同角度  ← user UNCHECKED to keep      │
├────────────────────────────────────────────────────────────────┤
│ Cluster #2  ...                                                 │
└────────────────────────────────────────────────────────────────┘
```

- Each cluster is collapsed by default; clicking the header expands.
- Each member has a checkbox; the **suggested_keep** image starts
  unchecked, every other member starts checked-for-delete.
- The header has a "全选/全不选" + a "保留全部" + a "复原 AI 建议" reset.
- Top sticky bar **共选中 N 张, 估计释放 X MB** + 「批量删除」(soft).
- Clicking 批量删除 opens an AlertDialog summarising N images, sample
  thumbnails (first 6), required confirmation before commit.
- Soft delete moves files + sidecars + annotations to
  `runs/_image_studio_trash/<date>/`. The Maintenance tab's existing
  trash-clear flow handles permanent deletion later.

Endpoints:

```
POST /api/image-studio/similarity/select
     {
       path, recursive?,
       layers: ["phash", "ai"],
       threshold?: float,                  -- minimum confidence
       strategy: "all_but_keep"            -- delete all members EXCEPT
                                           --   the suggested_keep
                | "all_high_confidence"    -- only clusters with conf >= X
                | "selected",              -- explicit member list below
       selected_paths?: [<path>]            -- when strategy=selected
     }
-> {
     selection: {
       cluster_id: [<path>, ...]            -- which paths in each cluster
     },
     total_count: 38,
     total_bytes: 432_000_000,
     sample_paths: [<first 6 for preview>],
     warnings: [...]                        -- e.g. "would delete a favourite"
   }

POST /api/image-studio/similarity/batch-delete
     { selection_token, paths_only?: bool }
                       (selection_token returned by /select; one-shot,
                        signed with the request body's hash to prevent
                        the UI sending a stale list)
-> {
     deleted: [<path>, ...],
     deleted_count, bytes_freed,
     trash_dir: "runs/_image_studio_trash/2026-05-17/...",
     errors: [{ path, reason }]
   }
```

Server safeguards before commit:

1. Every selected path must be under the same allowed root.
2. Refuse to delete a path that is **the only** member left in a
   cluster (defensive; UI shouldn't ever send this, but enforce anyway).
3. Refuse if `paths` includes any image with `favorite=1` unless the
   request body sets `force_favorites: true`.
4. Cap any single batch at 1000 images; larger batches must come in
   chunks (prevents a runaway click from nuking a folder).
5. Selection tokens expire after 60s server-side so the user can't
   page-stale a delete.

The endpoint is an explicit two-step (select → confirm) rather than a
one-shot DELETE. The first call is read-only and produces an
auditable summary; the dialog renders the summary; the second call
carries the token from the first. Mirrors how the existing
`/api/storage/archive/clear` flow gates destructive ops.

### 6.7 AI batch actions

Mirror the existing tagging session pattern: 202 + WebSocket / poll
status, persist terminal snapshots in `session_store` (table reused).

```
POST /api/image-studio/ai/caption
     {
       paths: [...],                 -- explicit list, OR
       path: "<dir>", recursive,
       filter: { ... }              -- same filter shape as /list
       task: "tagging.assist" |     -- which AI route
             "caption.rewrite",
       merge_strategy: "replace"   -- replace existing wd14 caption
                    | "append"     -- append AI sentence to caption
                    | "rewrite",   -- LLM rewrites existing caption
       max_concurrency: 4
     }
-> 202 { session_id }

GET /api/image-studio/ai/caption/{session_id}
-> standard tagging-session snapshot
```

Equivalent endpoints:
- `POST /api/image-studio/ai/quality` — VLM scores each image, writes
  `ai_quality_*` columns
- `POST /api/image-studio/ai/composition` — short composition description
  (角度 / 光照 / 主体)
- `POST /api/image-studio/ai/trigger-words` — given the dataset's
  characteristic, suggest trigger words and their natural-language
  template

### 6.8 Cost preview

```
POST /api/image-studio/ai/estimate
     { task, paths_count, image_resolution_avg }
-> { estimated_input_tokens, estimated_output_tokens, model_id, provider_id,
     estimated_usd_low, estimated_usd_high, breakdown: [...] }
```

Calls into `lorahub.core.ai.client.estimate_cost(task, n)` which uses
each provider's published per-token rates (table maintained alongside
the AI store) to produce a low/high band. Surfaced in the bulk-action
modal so the user sees "this will cost ~¥0.18" before clicking Run.

## 7. AI task wiring

The existing `AI_TASK_IDS` in `web/src/lib/api.ts` already includes
`tagging.assist`, `caption.rewrite`, `dataset.analyze`,
`training.diagnose`, `error.diagnose`, `global.default`. The studio
adds three more concrete uses:

### 7.1 `tagging.assist` — VLM caption single image

System prompt template (user-editable in Settings → 任务路由):

```
You are an annotator for LoRA training data. Look at the image and
produce a Danbooru-style caption: a single comma-separated list of
short tags. Begin with subject + count (1girl, 2boys, ...), then
relevant attributes (hair colour, eye colour, clothing), then the
scene/composition (close-up, full body, sitting, ...). 30-60 tags.
Do NOT add prose. Reply ONLY with the caption.

User content boundary:
<user_content>
... (only the image)
</user_content>
```

Request shape:
```ts
api.aiInvokeTask({
  taskId: "tagging.assist",
  prompt: "",              // empty; image carries the content
  systemPrompt: undefined, // route default
  // NB: we extend invoke to accept image data
})
```

**Schema extension** required: `AIInvokeTaskInput` gains an optional
`images?: { kind: "data_url" | "file_path", value: string }[]`. The
Python invoke layer reads each image, base64-encodes (if
file_path), and emits `{role:"user", content:[{type:"text",text:""},
{type:"image_url",image_url:{url:"data:image/..."}}]}`. Providers that
don't support vision return a 4xx from the upstream — surfaced as a
clear "selected model is text-only; pick a vision model on this task"
toast.

### 7.2 `caption.rewrite` — LLM rewrites existing caption

No image needed. Prompt template:

```
Rewrite the following Danbooru-style tag caption as a fluent natural
language sentence in {target_style: 中文 | English}, suitable as a
training prompt. Preserve all subject, attribute, and scene tags.

<user_content>
{caption_text}
</user_content>
```

### 7.3 `dataset.analyze` — survey statistics

Already wired for the dataset analyse button. Image Studio uses the
same task; the input is the studio's full directory listing summary
(top tags, resolution histogram, caption length stats). Output is a
markdown report rendered in a side drawer.

### 7.4 `quality.score` — new task id

Adds a 7th task id to `AI_TASK_IDS`: `quality.score`. System prompt:

```
Rate this training image on a 0–100 scale across:
- focus and sharpness
- prompt-fidelity (subject is clear, framing complete)
- LoRA suitability (one subject; not crop-cut)
Return JSON exactly:
{ "score": 0-100, "label": "优"|"中"|"差", "reason": "<≤30 words>" }
```

Routes through the user's preferred VLM. Result lands in
`ai_quality_*` columns and the inspector renders the badge.

### 7.5 `trigger.suggest` — new task id

Given the user's intent ("training a character LoRA of XXX") + a
random sample of 8 captions from the dataset, suggest:
- 1 trigger word (a unique token like `xxxchar`)
- 3 trigger templates ("a portrait of {trigger}, ...")

User can apply the trigger word as a caption prefix to all images in
one click via the bulk action runner.

### 7.6 `similarity.score` — new task id

Used by §6.5.2 mode B (pairwise judge). Single message body:

```
Compare two LoRA training images. Are they near-duplicates of the
same subject and scene? Reply EXACTLY one of:
- yes_strong   (same pose, framing, lighting; one is a near-recompress)
- yes_weak     (same subject + outfit but different pose/light/angle)
- no           (different subject OR materially different shot)
Then on a new line: a short reason in Chinese, ≤ 20 words.
```

The two images are sent as the user message's `image_url` parts
(see §7.1 schema extension). The response parser tolerates the model
prepending the verdict with quotes / punctuation.

### 7.7 `similarity.embed` — new task id (no chat)

Routed to a provider's `/v1/embeddings` endpoint, not chat completions.
Adds a small dispatcher branch in `lorahub/core/ai/client.py`:

```python
def embed(
    store: AIStore,
    *,
    provider_id: str,
    model_id: str,
    inputs: list[str | bytes],         # text or image-bytes (base64)
    timeout: float = 60.0,
) -> list[list[float]]:
    """POST <base>/v1/embeddings; returns one vector per input."""
```

Image embeddings are model-dependent — most OpenAI-compat providers
don't accept images on this endpoint, so we accept text descriptions
fallback: when the route is set to a text-only embedding model the
studio falls back to embedding each image's `ai_caption` (computed
in §7.1) instead of the pixels. This degrades gracefully and is
explicitly surfaced in the UI.

Result vectors land in `image_embeddings`, model id stamped so swapping
embedding providers re-clusters from scratch.

## 8. Frontend layout

Single React component tree under `web/src/pages/image-studio/`:

```
image-studio/
  index.tsx              ← page shell
  components/
    studio-toolbar.tsx
    filter-panel.tsx
    image-grid.tsx       (uses react-virtuoso for ≥1k image folders)
    image-tile.tsx
    inspector.tsx
    caption-diff-pane.tsx
    ai-bulk-modal.tsx    (caption / quality / dedupe / trigger flows)
    ai-cost-banner.tsx
    duplicates-tab.tsx
    pending-ops-bar.tsx  (sticky bottom: "3 项待应用 [应用] [放弃]")
  hooks/
    use-image-list.ts
    use-pending-ops.ts
    use-ai-batch.ts
  lib/
    filters.ts
    keyboard.ts          (j/k/x/e/q/d shortcuts)
```

Keyboard:
- `j` / `k` ↓ / ↑ in grid
- `space` open inspector / preview
- `x` toggle multi-select
- `e` open caption editor for current
- `q` open quality vote (1=优 / 2=中 / 3=差)
- `d` soft-delete (queues op; `Z` undoes)
- `Cmd/Ctrl-A` select all on current page
- `Cmd/Ctrl-Enter` apply all pending ops
- `?` help overlay

## 9. Implementation batches

To stay shippable, split into 5 commits each independently testable:

| # | Batch | Scope | Tests | Wall-clock |
|---|-------|-------|-------|------------|
| IS-0 | Foundation | new SQLite store, path allow-list reuse, image_annotations + image_pending_ops + image_phash tables, IAStore CRUD | 12 | 0.5 day |
| IS-1 | List + inspector | `/list` + `/image` routes, basic filter set (caption / aspect / quality), inspector panel, soft-delete | 18 | 1 day |
| IS-2 | Edit ops | rotate/flip/crop/replace/merge caption ops, pending-ops queue, apply pipeline (PIL re-encode), trash | 14 | 1 day |
| IS-3 | AI tasks (caption + quality) | extend invoke to carry images, two batch endpoints, session UI, cost preview | 20 | 1.5 days |
| IS-4 | Dedupe + trigger | phash64/dhash64 background scan, cluster API + UI, trigger-word task route | 14 | 1 day |
| IS-5 | Polish | virtualized grid (>1k images), keyboard shortcuts, help overlay, AI report drawer | — | 0.5 day |

Sum ≈ 5.5 days of focused work.

## 10. Acceptance criteria

The feature ships when:

1. Open a folder of 200 images → page renders within 2s on a 4090 box;
   thumbnails come in progressively from `/datasets/thumb`.
2. Filter caption=missing → only those rows show; counter updates.
3. Caption edit + close inspector + re-open → persisted to sidecar.
4. Apply VLM caption with merge=append on 30 images → progress bar +
   final state; sidecar files now end with the AI sentence.
5. Quality scoring on 100 images → all rows get a badge; failures
   listed as a per-image error in the session log.
6. Dedupe with phash64 threshold=8 → clusters listed; "keep this"
   radio commits a soft-delete on losers.
7. Apply rotate-right + crop 1:1 + replace_caption to one image →
   single Apply button mutates file in place, EXIF orientation reset.
8. SSH out, ls the folder, see the renamed files / removed sidecars
   exactly as the studio promised.
9. Disable the AI provider → bulk AI buttons render a clear "no AI
   route configured" empty state, caption-edit + manual ops still work.
10. Hit the page with a path NOT under any allowed root → 400 with
    a useful message ("not under any dataset root or job workspace").

## 11. Tests

### 11.1 Backend unit
- `IAStore` round-trip (annotations / pending_ops / phash) — incl.
  in-place updates, in-place soft delete, and the discovered-vs-manual
  distinction is preserved across rescans.
- `phash64` helper: hash known fixtures, threshold sanity.
- `apply_op` per op type with PIL stub.
- AI invoke extension: image part is correctly base64-encoded and
  routed when the route's model is vision-capable; clean error when
  not.

### 11.2 Backend HTTP
- list/image/annotations/ops/dedupe routes each have happy-path,
  invalid-path, missing-image cases.
- session pattern integration with the existing tagging session
  store (we reuse the `tagging` session-kind table to avoid
  schema sprawl; namespace via session metadata).

### 11.3 Frontend
- TypeScript only (no jsdom). The bar is `tsc -b` + `vite build` clean.
- Manual smoke checklist mirrors §10.

## 12. Risks & open questions

| risk | mitigation |
|------|------------|
| VLM token cost runs away on a 1000-image folder | Cost preview banner + provider rate-limit + per-call timeout; default `max_concurrency=4` |
| EXIF orientation rotates twice when we apply rotate after upstream auto-orient | Always read with `Pillow.ImageOps.exif_transpose` before applying ops, then strip orientation flag |
| Caption sidecar collision when the user is also editing in another tool | Optimistic lock via mtime; if mtime drift detected, surface "外部修改了 caption,接受/覆盖?" modal |
| Soft-deleted files clutter `runs/_image_studio_trash` forever | Maintenance tab gets a "清空回收站" alongside the existing archive cleanup |
| Dedupe scan on 5k images stalls UI | Background session pattern with checkpointed phash table; UI shows progress, partial results visible |
| Sidecar UTF-8 vs CRLF on Windows | Use `Path.write_text(text, encoding="utf-8", newline="")` to avoid CRLF doubling |

## 13. Future (post-1.0)

- AI-driven crop suggestion ("what's the best 1:1 crop of this image
  for face training?")
- Style consistency check across the dataset
- Caption diff-merge UI (3-way: original wd14, AI rewrite, user)
- Image2text retrieval ("find me other images in this dataset that
  match this prompt") via CLIP embeddings — single GPU job, run-once

---

## Appendix A — schema diff vs current AI subsystem

`AI_TASK_IDS` is extended (frontend + backend `_LORAHUB_TASKS` seed):

```
const AI_TASK_IDS = [
  "global.default",
  "tagging.assist",
  "caption.rewrite",
  "dataset.analyze",
  "training.diagnose",
  "error.diagnose",
  "quality.score",     // NEW
  "trigger.suggest",   // NEW
] as const
```

`AIInvokeTaskInput` gains:

```ts
images?: Array<{ kind: "data_url" | "file_path"; value: string }>
```

Python `invoke` resolves `file_path` entries into base64 data URLs
using the Pillow `IMAGE_SUFFIXES` set; rejects files outside the
allow-list with the same `dataset_files` policy.

## Appendix B — folder tree after IS-5

```
lorahub/
  api/
    routers/image_studio.py          NEW
    image_studio_store.py            NEW
    image_studio_ops.py              NEW (PIL apply pipeline)
    image_studio_phash.py            NEW
  core/
    ai/client.py                     touched (image part support)
web/src/pages/
  image-studio/                      NEW (see §8)
runs/
  image_studio.sqlite                NEW (per-user)
  _image_studio_trash/<date>/        NEW (soft-delete bucket)
  .thumbs/                           reused
tests/
  test_image_studio_store.py         NEW
  test_image_studio_router.py        NEW
  test_image_studio_ops.py           NEW
  test_image_studio_phash.py         NEW
docs/
  image-studio.md                    NEW (this file's user-facing slice)
```
