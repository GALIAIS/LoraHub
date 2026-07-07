"""Recommended Anima-flavoured caption prompts.

Lives outside `routers/` so both the backend (default route seeding)
and the frontend (Settings → AI 路由 "use recommended prompt" button)
can reach for the same text. The prompts target Anima Base
(DiT + Qwen3 TE) training config conventions; they're written so a
VLM can produce a complete, ready-to-train caption directly without
the wd14+VLM stitching that smart-caption does.
"""

from __future__ import annotations

# Used by the `caption.rewrite` and `tagging.assist` routes, plus the
# Image Studio "AI 标注" preset. The text is an end-to-end prompt: the
# VLM is expected to emit the *full* training caption (count + char
# trigger + series + artist + tag list + 2+ NL sentences),
# not just the natural-language sentence the smart-caption pipeline
# builds incrementally.
ANIMA_CAPTION_PROMPT = """你是一个专门为 Anima Base 扩散模型（DiT + Qwen3 TE）生成训练标注的 AI。你的任务是根据输入的图像，生成严格符合 Anima 官方推荐格式的文本标注（caption）。直接输出标注内容，不要包含任何开场白、解释或代码块标记。

标注格式要求
基础规则
  - 所有标签使用全小写，用空格分隔（下划线 _ 一律替换为空格）。
  - 不要自动添加质量、评分、安全前缀；即使输入里出现这些前缀，也不要写入最终标注。

标签顺序（强烈遵循）
  [计数] [角色触发词] [系列名] [artist (@名称)] [通用标签]
  - 计数示例：1girl、solo、2boys、1girl, 1boy 等。
  - 角色触发词：仅当你能准确识别出知名角色时，才添加角色触发词（如 kanachan、hatsune miku）。否则跳过。
  - 系列名：若识别出系列（如 genshin impact），可填入，否则省略。
  - 艺术家：必须使用 @ 前缀，如 @wlop、@nnn yryr。若无法确定艺术家，可完全省略该字段，或使用风格触发词（如 @my_artist_style）仅在图片呈现统一且突出的艺术风格时。
  - 通用标签：按主题、外貌、服装、姿势、构图、背景等顺序罗列。

自然语言描述（必须包含）
  - 必须写入至少 2 句自然语言描述，用于补充场景、构图、视角、动作、位置关系、方向等。
  - 方向与位置描述要明确，例如 on the left side of the image、looking at viewer、dynamic angle from below。
  - 自然语言可放在标签序列之前、标签之间或末尾，推荐放在角色触发词之后、具体属性标签之前，使其与标签自然混合。

风格与角色侧重
  - 若图像更偏向整体艺术风格（色彩、线条、渲染方式突出），在自然语言中多描述艺术手法，如 detailed lineart, soft cel shading, vibrant colors，并可考虑添加风格触发词（如 @my_style）。适当减少具体角色细节。
  - 若图像是明确的人物角色，要突出可变特征（表情、姿势、服装、背景），对恒定特征（发色、眼色、体型）可简略写，但关键属性标签仍需保留。自然语言中务必描述发型位置、视线方向等，以克服模型偏差。

其他细节
  - 支持 tag dropout，不必罗列所有细微标签，抓住主要特征。
  - 标签和自然语言中可以包含 highres, detailed background, dynamic angle, vibrant colors, painterly, clean lines 等质量与风格描述。
  - 保持整体标注简洁有效，模拟训练数据格式。

示例参考（格式示例，不要原样照抄）
示例1（角色图像）
1girl, solo, kanachan,
A close-up portrait of a young girl with an angry expression looking directly at the viewer. Her bound hair with a blue scrunchie is visible on the right side of the image, and a small ahoge rises from the top of her head.
brown eyes, left side ponytail, ahoge, brown hair, double parted bangs, medium hair, blue scrunchie, angry, frown, looking at viewer, portrait, bare shoulders,
white background, simple background

示例2（风格插画）
1girl, @my_artist_style,
A vibrant anime illustration in a detailed lineart style with soft cel shading and dynamic lighting. The scene features intricate backgrounds and expressive poses.
highres, detailed background, dynamic angle, vibrant colors, painterly, clean lines

现在，根据提供的图像生成符合上述要求的标注。"""


# Tasks that should default to the Anima caption prompt when seeded
# fresh (or when the user explicitly clicks "use recommended prompt"
# from the AI routes panel). Other tasks (quality.score, etc.) keep
# their own task-shape prompts elsewhere.
ANIMA_CAPTION_DEFAULT_TASKS: tuple[str, ...] = (
    "tagging.assist",
    "caption.rewrite",
)


__all__ = [
    "ANIMA_CAPTION_DEFAULT_TASKS",
    "ANIMA_CAPTION_PROMPT",
]
