# LoraHub Linear Workbench 主题设计规格

**状态：** 已实现

**最后更新：** 2026-07-13

**适用范围：** `web/` 全部应用壳层、shadcn 基础组件、业务页面与图表视觉

**主题标识：** `data-ui-style="linear"`

## 1. 目标

本主题以登录后的 Linear 工作区为视觉研究对象，但不是逐像素复制。目标是把其成熟的工作台设计方法迁移到 LoraHub：

- 内容优先，侧栏、边框与非活动控件主动后退。
- 保留高信息密度，同时通过排版、留白和表面层级降低拥挤感。
- 浅色和深色使用同一套语义 token，不依赖页面内硬编码颜色。
- 静态页面几乎不使用阴影；只在 Dialog、Popover、Dropdown 等浮层建立高度感。
- 训练状态、GPU、错误和图表仍保持 LoraHub 所需的多语义颜色，不把产品做成单色皮肤。
- 复用现有 shadcn/Base UI 组件和 `.shiro-*` 结构钩子，不复制一套平行业务组件。

新主题替换旧的第二套主题。Shiro 保留，主题选择改为 `Shiro / Linear`。

## 2. 实际界面审查结论

本规格基于 2026-07-13 对 `https://linear.app/galiais` 登录后界面的直接检查，覆盖工作区列表、Issue 详情、设置页、创建弹层以及浅色/深色模式。检查只用于观察视觉结构，没有读取或记录认证信息。

### 2.1 应用壳层

| 观察 | 结论 | LoraHub 映射 |
| --- | --- | --- |
| 侧栏与浏览器底色接近，主内容是轻微抬高的大画布 | 主次区域靠明度和边界区分，不靠卡片堆叠 | Sidebar 保持扁平，`SidebarInset` 成为主画布 |
| 桌面主画布有约 12px 圆角和极轻边线 | 圆角只用于大结构边界 | 页面内小面板统一 6-8px，不层层放大圆角 |
| 顶栏与主画布连成一体 | 工具栏不是独立浮卡 | 去掉工具栏渐变和明显阴影 |
| 浅色侧栏比主画布略灰，深色侧栏比主画布略暗 | 同一层级只需小幅明度差 | `sidebar` 与 `background` 保持 2-4% 明度差 |

### 2.2 导航与列表

| 观察 | 结论 | LoraHub 映射 |
| --- | --- | --- |
| 非活动导航文字和图标明显降对比 | 侧栏不与内容争夺注意力 | inactive 使用 `sidebar-muted-foreground` |
| 当前导航是中性浅底，不使用高饱和色块或左侧彩条 | 选中态依靠低对比填充 | active 为 `sidebar-accent` + 主文字 |
| 列表行主要靠 1px 分隔和 hover 底色 | 重复信息应使用行，不使用独立卡片 | Jobs、Configs、Artifacts 优先行式布局 |
| 状态、ID、日期比标题更弱 | 字色和字号承担层级 | 主标题 13px/500，元数据 12px/400 |

### 2.3 设置与详情

| 观察 | 结论 | LoraHub 映射 |
| --- | --- | --- |
| 设置内容限制在约 640-760px 可读宽 | 长表单不应铺满超宽屏 | 设置与配置表单使用受控内容宽度 |
| 相关字段放在一个分组表面内，字段之间用细线 | 一个分组一层表面，避免卡片套卡片 | `Card` 仅作为字段组，不包装每个字段 |
| 详情右栏用少量分组面板组织属性 | 侧栏适合紧凑属性而非大标题 | 训练详情侧栏采用 12px 标签和 13px 值 |
| 开关无投影，关闭态中性，开启态仅轨道着色 | 控件状态要清楚但克制 | Switch 全部去除阴影，统一 28px 轨道体系 |

### 2.4 浮层

| 观察 | 结论 | LoraHub 映射 |
| --- | --- | --- |
| 创建弹层比页面表面亮一级 | 浮层才需要明确 elevation | Popover/Dialog 使用 `popover` 和专用阴影 |
| 遮罩降低背景对比但不完全盖黑 | 保留上下文 | overlay 深色 48%，浅色 24% |
| 操作标签紧凑、接近圆角胶囊 | 胶囊只适合过滤器和短属性 | 普通按钮保持 6px，Badge/Filter 可 full |

### 2.5 字体与实际色阶

Linear 当前使用 Inter Variable 系列。LoraHub 已内置 Geist Variable，两者均适合高密度工具界面，因此不新增字体依赖，继续使用 Geist，并针对中文保留现有字体回退。

实际检查到的 Linear 表面接近：

- 浅色基础层约 `lch(95.94 0.5 282)`，主画布约 `lch(98.94 0.5 282)`。
- 深色主画布约 `lch(4.52 0.3 272)`，边界和卡片只比其亮少量。
- 页面正文基准为 16px，但导航、元数据和控件普遍落在 12-14px。
- 浮层和主画布可使用 12px 大结构圆角，普通控件不跟随放大。

实现使用 OKLCH 重新标定，不直接复制来源站点数值。

## 3. 设计原则

### 3.1 结构应被感知，而不是被描边

- 页面区块优先靠对齐、间距和背景层级分组。
- 同一容器内只保留必要分隔线。
- 禁止每个区块都带独立背景、边框和阴影。
- 卡片不得嵌套卡片；内部需要分组时使用 `Separator`、标题或弱底色行。

### 3.2 密度由一致尺寸保证

- 控件高度只使用 24 / 28 / 32 / 36px 四档。
- 相同层级的标题、字段和列表行保持同一垂直节奏。
- 日志、数据表和训练列表可以高密度；图像画廊和复杂表单允许更松。
- 不通过缩小到 11px 以下来换取密度。

### 3.3 强调色只表示动作和焦点

- 主强调色用于主操作、焦点、链接和选中控件。
- 导航选中态默认使用中性填充，不铺强调色。
- 成功、警告、错误、运行中继续使用独立语义色。
- 图表允许多序列颜色，但色板必须稳定、可区分、适配深浅色。

### 3.4 动效服务状态变化

- Hover 和 focus 只做颜色、边框和轻微透明度过渡。
- 页面切换和主题切换不做大幅位移、弹跳或光晕。
- 进度、数值和展开动画继续由 Anime.js 管理；主题样式不再引入另一套动画机制。
- `prefers-reduced-motion` 必须停用非必要动画。

## 4. 主题架构

### 4.1 模式

```ts
type ThemeMode = "light" | "dark" | "system"
type StyleMode = "shiro" | "linear"
```

- `ThemeMode` 决定明暗。
- `StyleMode` 决定视觉语言。
- 根节点组合为 `.dark[data-ui-style="linear"]`。
- 新存储键使用 `lorahub.ui.style.v3`，避免旧主题值继续生效。
- 未识别值回退到 `shiro`，不允许产生半套主题。

### 4.2 实现边界

主题通过三层完成：

1. **语义 token：** 覆盖 shadcn 现有 `background/card/popover/primary/...`。
2. **产品 token：** 覆盖 `surface/control/state/chart/shell` 等 LoraHub 变量。
3. **组件配方：** 使用 `data-slot` 与现有 `.shiro-*` 钩子统一 Sidebar、Card、Button、Input、Tabs、Dialog、Table 等外观。

不建立 `LinearButton`、`LinearCard` 等分叉组件。业务组件只消费语义 token。

## 5. 色彩系统

### 5.1 浅色 token

| Token | 建议值 | 用途 |
| --- | --- | --- |
| `--background` | `oklch(0.972 0.003 285)` | 应用基础层 |
| `--foreground` | `oklch(0.205 0.008 285)` | 主文字 |
| `--card` | `oklch(0.995 0.002 285)` | 主画布、分组表面 |
| `--popover` | `oklch(1 0 0)` | 浮层 |
| `--secondary` | `oklch(0.952 0.004 285)` | 次级控件 |
| `--muted` | `oklch(0.955 0.003 285)` | 弱底色 |
| `--muted-foreground` | `oklch(0.49 0.01 285)` | 次要信息 |
| `--accent` | `oklch(0.925 0.008 285)` | hover / selected 中性底 |
| `--primary` | `oklch(0.57 0.18 278)` | 主操作与焦点 |
| `--border` | `oklch(0.885 0.005 285)` | 默认边界 |
| `--input` | `oklch(0.86 0.006 285)` | 控件边界 |
| `--ring` | `oklch(0.61 0.17 278)` | focus ring |
| `--sidebar` | `oklch(0.952 0.003 285)` | 侧栏 |
| `--sidebar-accent` | `oklch(0.895 0.005 285)` | 当前导航 |

浅色主题禁止纯白铺满整个窗口。纯白只用于主画布或浮层，窗口底色保留很淡的暖中性灰。

### 5.2 深色 token

| Token | 建议值 | 用途 |
| --- | --- | --- |
| `--background` | `oklch(0.135 0.004 285)` | 应用基础层 |
| `--foreground` | `oklch(0.94 0.004 285)` | 主文字 |
| `--card` | `oklch(0.162 0.004 285)` | 主画布、静态表面 |
| `--popover` | `oklch(0.205 0.006 285)` | 浮层 |
| `--secondary` | `oklch(0.205 0.005 285)` | 次级控件 |
| `--muted` | `oklch(0.19 0.005 285)` | 弱底色 |
| `--muted-foreground` | `oklch(0.68 0.008 285)` | 次要信息 |
| `--accent` | `oklch(0.235 0.006 285)` | hover / selected 中性底 |
| `--primary` | `oklch(0.72 0.14 278)` | 主操作与焦点 |
| `--border` | `oklch(0.265 0.006 285)` | 默认边界 |
| `--input` | `oklch(0.29 0.007 285)` | 控件边界 |
| `--ring` | `oklch(0.72 0.14 278)` | focus ring |
| `--sidebar` | `oklch(0.115 0.003 285)` | 侧栏 |
| `--sidebar-accent` | `oklch(0.225 0.006 285)` | 当前导航 |

深色不使用蓝黑或大面积紫色。中性基底仅带极低的 285° 色相，避免纯黑生硬，也避免单一深蓝主题。

### 5.3 语义色

| 语义 | 浅色 | 深色 | 使用限制 |
| --- | --- | --- | --- |
| Running / info | 稳定蓝 | 明亮蓝 | 状态点、Badge、图表 |
| Success | 中深绿 | 明亮绿 | 完成、健康 |
| Warning | 琥珀 | 偏亮琥珀 | 风险、等待 |
| Danger | 中深红 | 明亮红 | 错误、破坏操作 |
| Diagnostic | 紫 | 淡紫 | AI 分析、诊断事件 |

状态色面积原则：大面积表面只加入 4-8% 混色；文字和图标可使用完整语义色。

### 5.4 图表色板

```css
--chart-1: blue;
--chart-2: teal;
--chart-3: violet;
--chart-4: amber;
--chart-5: red;
```

- 原始 loss 使用低透明细线或点云，平滑线使用主色实线。
- 不同训练 run 必须使用稳定的独立颜色，不能只改变透明度。
- 网格线只比背景高一个层级，轴标签使用 muted foreground。
- Tooltip 使用 `popover`，数字使用 tabular nums。
- 主题只调整视觉 token，不改变图表降采样、平滑和缩放逻辑。

## 6. 排版

### 6.1 字体

```css
--font-sans: 'Geist Variable', 'Microsoft YaHei UI', 'PingFang SC',
  'Noto Sans CJK SC', system-ui, sans-serif;
--font-mono: ui-monospace, 'Cascadia Code', 'SFMono-Regular', Consolas, monospace;
```

不新增字体包。日志和数值使用 mono；正文、表单和导航使用 Geist。

### 6.2 字号与行高

| 角色 | 字号 | 字重 | 行高 |
| --- | --- | --- | --- |
| 页面标题 | 16px | 600 | 24px |
| 面板标题 | 13px | 600 | 20px |
| 正文/字段值 | 13px | 400-450 | 20px |
| 导航项 | 13px | 450 | 20px |
| 次要信息 | 12px | 400 | 18px |
| 分组标签 | 11px | 500 | 16px |
| 日志 | 12px | 400 | 18px |
| KPI | 20-24px | 550 | 1.1 |

规则：

- `letter-spacing: 0`；不使用负 tracking。
- 中文正文最小 12px，交互文本最小 12px。
- ID、PID、step、loss、显存等使用 `font-variant-numeric: tabular-nums`。
- 标题通过字重和前后间距建立层级，不使用超大字号。

## 7. 间距、尺寸与圆角

### 7.1 间距

使用 4px 基线：`4 / 8 / 12 / 16 / 20 / 24 / 32`。

| 场景 | 规格 |
| --- | --- |
| 紧凑图标与文字 | 6px |
| 列表行左右 padding | 12px |
| 普通面板 padding | 16px |
| 页面主区 gap | 16px |
| 大章节间距 | 24px |

### 7.2 稳定高度

| 组件 | 高度 |
| --- | --- |
| Icon button xs / sm | 24 / 28px |
| Sidebar item | 30px |
| Button / Input sm | 28px |
| Button / Input md | 32px |
| 表单主输入 | 36px |
| Tab / segmented item | 28-30px |
| 单行列表 | 36-40px |
| 双行任务列表 | 48-52px |
| Top toolbar | 44-48px |

### 7.3 圆角

| Token | 值 | 场景 |
| --- | --- | --- |
| `--radius-sm` | 4px | 小标签、紧凑控制 |
| `--radius-md` | 6px | Button、Input、导航项 |
| `--radius-lg` | 8px | Card、Popover |
| `--radius-xl` | 12px | 主画布、Dialog |
| full | 9999px | 状态点、头像、过滤器胶囊 |

避免所有元素共享同一个圆角。嵌套容器的内圆角必须小于外圆角。

## 8. 表面与阴影

### 8.1 表面层级

1. `background`：窗口底色和 Sidebar 基础。
2. `card`：主画布与静态字段组。
3. `muted`：hover、行选择、辅助条。
4. `popover`：Dropdown、Popover、Tooltip、Dialog。

### 8.2 阴影配方

- 主画布：最多 `0 1px 1px rgba(..., .04)`，可只用边框。
- Card、Table、Sidebar：无阴影。
- Popover：`0 8px 24px -12px` + 1px 边框。
- Dialog：`0 18px 48px -20px` + 1px 边框。
- Hover：不抬升，不改变尺寸，只改变背景或边框。

禁止渐变表面、内发光、高光描边和装饰性网格背景。

## 9. shadcn 组件规格

### 9.1 Button

- `default`：主色实底，只用于当前区域的主动作。
- `secondary`：中性弱底，无阴影。
- `outline`：透明底 + `border`。
- `ghost`：透明，hover 使用 `accent`。
- `destructive`：仅删除、强制终止等不可逆操作。
- 图标按钮固定正方形，使用 Lucide 图标和 Tooltip。
- active 只降低亮度，不做位移或缩放。

### 9.2 Input / Textarea / Select

- 默认表面与所在容器一致，边界清晰但低对比。
- focus 为 1px ring + 1px 外扩，不改变控件宽高。
- 错误状态同时显示边框、图标或错误文案，不能只变红。
- Select trigger、Input、数字输入保持同高、同圆角、同 padding。
- Placeholder 比正文低一个层级，不使用过浅灰。

### 9.3 Tabs 与过滤器

- 页面主视图 Tabs 使用文字 + 底部指示，不使用大胶囊。
- `全部 / 运行中 / 失败` 等短模式筛选使用紧凑 segmented pills。
- 同一区域不同时出现两套视觉相同的 Tabs。
- 选中态不使用阴影。

### 9.4 Card / Settings group

- Card 只用于独立对象、字段组、重复实体或真正需要框定的工具。
- 设置项采用一张分组表面，字段行之间使用 Separator。
- Card 内禁止再放视觉上完整的 Card。
- Header 标题 13px，Description 12px，Content 保持 12-16px padding。

### 9.5 Sidebar

- 展开宽度沿用当前信息架构，导航项高度 30px。
- 分组标题 11px muted，不使用大写宽字距。
- active 为中性底 + 主文字，不显示左侧彩色竖线。
- 未激活图标和文字使用同一 muted 层级。
- Footer 主题选择器使用两个分段按钮：`Shiro / Linear`。

### 9.6 Dialog / Sheet / Popover / Dropdown

- Dialog 必须有标题；关闭按钮位于右上角。
- 弹层比页面亮/暗一个层级，避免与底层融为一体。
- 菜单项高度 30-32px，快捷键右对齐 mono 11px。
- 危险菜单项只有在 hover/focus 时显示弱危险底。
- Sheet 在移动端全宽或接近全宽，桌面端保持受控宽度。

### 9.7 Table / List / Pagination

- 表头 11-12px、500；内容 12-13px。
- 行 hover 使用 3-5% 明度变化。
- selected 使用中性 accent，不大面积铺主色。
- Pagination 的页码、上一页、下一页必须同高；文字不得撑破容器。
- 长路径和任务 ID 使用中间或尾部截断，完整值放 Tooltip。

### 9.8 Badge / Progress / Switch

- Badge 高 20-22px；状态使用图标/点 + 文字。
- Progress 轨道无内阴影，填充变化平滑，不在末端添加装饰图标。
- Switch 完全去除 box-shadow；开启态使用 primary，关闭态使用 input。
- 二值控件使用 Switch/Checkbox，不使用文字按钮模拟。

### 9.9 Toast / Alert / Empty / Loading

- Toast 宽度受控，错误内容可换行，不使用高饱和整块背景。
- Alert 用左侧语义图标和弱底色，不使用粗重边框。
- Empty 只包含短原因和主要动作，不放大插画。
- Loading 优先使用结构匹配的 Skeleton；长任务显示阶段和可取消状态。

## 10. LoraHub 页面映射

### 10.1 数据面板

- 顶部状态条保持单行、低对比底色，GPU/CPU/内存使用稳定列宽。
- 主机、CPU、GPU 信息使用同一基线和底边对齐。
- 数据面板不新增装饰卡；重点数值通过字重和语义色突出。

### 10.2 训练任务

- 左侧列表按行组织，状态、任务名、时间形成三层文字。
- 详情内容使用主画布，不把每个指标做成浮卡。
- 原始日志使用专用高对比控制台表面，但仍消费主题 token。
- 生命周期按钮按危险程度排序，强制终止必须是 destructive。

### 10.3 训练分析

- KPI 区可使用分组表面，但不做独立彩色卡片矩阵。
- 图表背景与主画布同级，靠标题和坐标区域分组。
- raw loss、平滑线、验证线和参考 run 必须通过图例和颜色明确区分。
- 无 val/epoch 的后端显示原因，不让空图直接消失。

### 10.4 图像工作台

- 阶段导航使用紧凑 Tabs；主要打标入口保持首屏可见。
- 图片网格保留较松间距，工具栏和批量操作保持紧凑。
- 提示词库、Caption 类型和触发词控件必须同高对齐并可响应缩放。

### 10.5 配置表单与设置

- 内容最大宽度 820px；复杂表单可以使用双列，但移动端必须单列。
- 后端专属字段按条件出现，不显示无关字段占位。
- 折叠组标题保持轻量，展开后不产生卡片套卡片。
- 字段错误紧邻控件，明确显示字段名与修复方向。

## 11. 响应式规则

| 断点 | 行为 |
| --- | --- |
| `< 768px` | Sidebar 变 Drawer；主画布取消外部圆角；详情全屏；工具条允许两行 |
| `768-1279px` | Sidebar + 主画布；右侧属性使用 Sheet 或可折叠栏 |
| `>= 1280px` | Sidebar + 主区 + 可选详情栏；设置内容限制可读宽度 |

- 移动端点击目标不小于 36px。
- 桌面紧凑控件可为 28-32px，但必须有可见 focus。
- 固定工具条和底栏不得遮挡局部滚动内容。
- 任何按钮文本、状态或分页数字不得越界。

## 12. 动效规范

| 场景 | 时长 | 缓动 |
| --- | --- | --- |
| Hover / focus 色彩 | 120-150ms | ease-out |
| Popover / Dropdown | 140-180ms | ease-out |
| Dialog / Sheet | 160-220ms | ease-out |
| Collapsible | 180-240ms | cubic-bezier(0.22, 1, 0.36, 1) |
| 数值与进度 | 320-520ms | cubic-bezier(0.22, 1, 0.36, 1) |
| 主题切换 | 180-240ms | cross-fade |

Linear 主题不使用大范围圆形揭示作为默认主题切换效果。Shiro 可保留自身特色，但二者必须支持 reduced motion。

## 13. 可访问性

- 正文对比度至少 4.5:1，大文字和非文本边界至少 3:1。
- `:focus-visible` 始终可见，不能只依赖 hover。
- 状态不能只靠颜色；必须配文字、点或图标。
- Dialog、Sheet、Drawer 必须有可访问标题。
- 日志滚动不得抢焦点；自动跟随必须可暂停。
- 动画、进度和实时数据需兼容 reduced motion。
- 浅色、深色、Windows 高缩放和中文长文案必须单独检查。

## 14. 实施计划

### Phase 1：主题身份与 token

1. `StyleMode` 改为 `shiro | linear`。
2. 使用新的 v3 存储键，删除旧主题入口和选择文案。
3. 在 `index.css` 建立 Linear 浅色/深色语义 token。
4. 删除旧主题变量、选择器和装饰性渐变配方。

### Phase 2：应用壳层与基础组件

1. 重做 Sidebar、SidebarInset、Toolbar 的层级。
2. 统一 Button、Input、Select、Tabs、Card、Dialog、Popover、Table、Badge、Progress、Switch。
3. 调整 focus、disabled、hover、selected、error 状态。
4. 保持 shadcn 组件 API 不变，避免业务页面大面积改写。

### Phase 3：高频业务页面校准

1. 数据面板与全局状态条。
2. 训练任务、训练详情和原始日志。
3. 训练分析图表与 KPI。
4. 图像工作台与提示词库。
5. 配置表单、设置和维护更新。

### Phase 4：验证

1. TypeScript 构建。
2. 浏览器检查浅色/深色各一轮。
3. 桌面 1440×900、宽屏 1920×1080、移动端 390×844。
4. 检查 Dialog、Dropdown、Toast、表格溢出、局部滚动和键盘 focus。
5. 对比主题切换前后布局尺寸，确保仅样式变化不会造成内容跳动。

## 15. 验收标准

- [ ] 主题选择中只存在 `Shiro / Linear`。
- [ ] 代码和 CSS 中不存在旧主题 token 或旧选择器。
- [ ] Linear 浅色与深色均有完整语义 token，不靠页面硬编码修补。
- [ ] Sidebar inactive 内容明显后退，active 不使用彩色竖条。
- [ ] 主画布、Toolbar、Card 不使用渐变或装饰网格。
- [ ] 静态 Card、Table、Sidebar 无明显阴影。
- [ ] Button、Input、Select、Pagination 高度和圆角一致。
- [ ] Switch 与 Progress 无阴影和末端装饰。
- [ ] Dialog、Popover、Toast 的 elevation 明确且不过重。
- [ ] 图表在深浅色下保留原始数据、平滑线和多序列辨识度。
- [ ] 桌面、移动端、中文长文案和 125%-200% 缩放不溢出。
- [ ] 键盘焦点、错误、disabled、loading、empty 状态完整。
- [ ] 不复制 Linear 商标、专有图标、文案或整页布局。

## 16. 非目标

- 不改变 LoraHub 的信息架构和路由。
- 不把页面功能改造成 Issue 管理器。
- 不引入新的 CSS-in-JS、主题运行时或组件库。
- 不复制 Linear 的品牌资源、快捷键体系或专有交互。
- 不在本轮重写图表数据处理、训练状态机或后端 API。

## 17. 参考

- 登录后的 `https://linear.app/galiais` 工作区：仅作本次视觉观察。
- Linear 官方文章：[Behind the latest design refresh](https://linear.app/now/behind-the-latest-design-refresh)
- Linear 官方文章：[How we redesigned the Linear UI](https://linear.app/now/how-we-redesigned-the-linear-ui)
- LoraHub 当前 shadcn/Base UI 组件与 `web/src/index.css` 语义 token。

## 18. 一句话实现原则

> 用温和的中性表面、克制的主色、统一的紧凑尺寸和明确的浮层高度，让 LoraHub 的训练信息成为视觉中心，而不是让主题本身成为视觉中心。
