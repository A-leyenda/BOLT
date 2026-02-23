from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from PIL import Image

from bolt.utils.io import iter_jsonl, safe_open_image


@dataclass
class KDSample:
    image: str
    question: str
    options: List[str]
    teacher_probs: List[float]
    gt_idx: Optional[int]
    answer: str = ""


def iter_kd_samples(kd_jsonl: str | Path) -> Generator[KDSample, None, None]:
    for r in iter_jsonl(kd_jsonl):
        opts = r.get("options")
        probs = r.get("teacher_probs")
        if not isinstance(opts, list) or not opts:
            continue
        if not isinstance(probs, list) or len(probs) != len(opts):
            continue
        yield KDSample(
            image=str(r.get("image", "")),
            question=str(r.get("question", "")),
            options=[str(o) for o in opts],
            teacher_probs=[float(x) for x in probs],
            gt_idx=r.get("gt_idx", None),
            answer=str(r.get("answer", "")),
        )
