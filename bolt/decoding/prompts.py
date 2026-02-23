from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


DEFAULT_DESC_PROMPT = (
    "Describe the WHOLE image concisely in English and Chinese. "
    "Mention: scene layout, key objects, robot gripper (open/closed), current phase "
    "(approach|pre_grasp_align|grasp_closing|firmly_grasping|lift|move|place|release), "
    "and any visible COLORED POINTS or ARROWS if present, plus a plausible instruction. "
    'Return JSON ONLY: {"en":"<3-6 sentences>", "zh":"<对应的3-6句中文描述>"}'
)

DEFAULT_CONSTRAIN_TEMPLATE = (
    "Answer the question by choosing EXACTLY ONE option from the set below.\n"
    "Options: {opts}\n"
    "Rules: output EXACTLY the chosen option text; no punctuation, no explanation.\n"
    "Q: {q}\n"
    "A:"
)

DEFAULT_QD_GUIDELINE = (
    "You are solving a constrained multiple-choice robot VQA question.\n"
    "First, write 2-4 short verification sub-questions (no more than 10 words each) "
    "that help disambiguate the correct option.\n"
    "Then answer the original question by choosing exactly one option.\n"
)

def format_constrained_prompt(question: str, options: List[str], template: str = DEFAULT_CONSTRAIN_TEMPLATE) -> str:
    return template.format(opts=", ".join(options), q=question.strip())


def format_qd_preamble(
    type_key: str,
    question: str,
    options: List[str],
    desc: Optional[Dict[str, str]] = None,
    retrieved_examples_block: str = "",
    guideline: str = DEFAULT_QD_GUIDELINE,
) -> str:
    d_en = (desc or {}).get("en", "")
    d_zh = (desc or {}).get("zh", "")
    parts = [guideline]
    if d_en or d_zh:
        parts.append(f"[Image description]\nEN: {d_en}\nZH: {d_zh}\n")
    if retrieved_examples_block:
        parts.append("[Retrieved examples]\n" + retrieved_examples_block + "\n")
    parts.append(f"[Question type] {type_key}\n")
    return "\n".join(parts).strip()
