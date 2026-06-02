"""Multimodal grounding for cmbagent_lg.

cmbagent_lg is otherwise text-only: agents see code + stdout + a file-path
manifest, but not the *contents* of the plots they produce. That let the
researcher write a report whose conclusions contradicted the figures. When
`PlanContext.vlm_enabled` is set, this module hands the actual generated PNGs to
a vision-capable agent (the researcher, and the step evaluator), so it can read
trends straight off the plots instead of inferring them from code alone.

Provider-agnostic: it emits the LangChain multimodal content-block format
(`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`),
which ChatGoogleGenerativeAI, ChatOpenAI and ChatAnthropic all accept — so it
works with any modern vision model (gemini-*, gpt-4o/gpt-5*, claude-*).
"""

from cmbagent_lg.vlm.images import collect_images, images_from_manifest, with_images
from cmbagent_lg.vlm.reviewer import image_reviewer, route_after_image_reviewer

__all__ = [
    "collect_images",
    "images_from_manifest",
    "with_images",
    "image_reviewer",
    "route_after_image_reviewer",
]
