from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from PIL import Image
from tqdm import tqdm

from bolt.data.robo2vlm import iter_examples_any, normalize_example
from bolt.decoding.option_scoring import OptionScorer
from bolt.utils.io import ensure_dir, safe_open_image
from bolt.utils.text import match_option_index


@dataclass
class TeacherCacheConfig:
    data_path: str
    image_root: str
    out_jsonl: str
    tau_kd: float = 2.0
    short_edge: Optional[int] = None
    max_samples: int = -1


def build_teacher_cache(
    scorer: OptionScorer,
    cfg: TeacherCacheConfig,
) -> None:
    out_path = Path(cfg.out_jsonl)
    ensure_dir(out_path.parent)

    n_written = 0
    with out_path.open("w", encoding="utf-8") as w:
        for raw in tqdm(iter_examples_any(cfg.data_path), desc="Teacher cache"):
            ex = normalize_example(raw)
            if ex is None:
                continue
            opts = ex.get("options") or []
            if not opts:
                continue

            img_path = Path(cfg.image_root) / ex["image"]
            if not img_path.exists():
                continue
            img = safe_open_image(img_path)

            prompt_text = ex["question"]
            # IMPORTANT: the constrained prompt should include options
            # to match the paper setup. We store the raw question here and build prompts later.
            # (For caching we score directly using the constrained prompt.)
            from bolt.decoding.prompts import format_constrained_prompt
            constrain_prompt = format_constrained_prompt(ex["question"], opts)

            scores_obj = scorer.score_options(img, constrain_prompt, opts, short_edge=cfg.short_edge, tau=cfg.tau_kd)
            gt_idx = match_option_index(opts, ex.get("answer", ""))

            row = {
                "image": ex["image"],
                "question": ex["question"],
                "options": opts,
                "answer": ex.get("answer", ""),
                "gt_idx": gt_idx,
                "teacher_scores": scores_obj.scores,
                "teacher_probs": scores_obj.probs,
            }
            if "id" in ex:
                row["id"] = ex["id"]
            if "type" in ex:
                row["type"] = ex["type"]

            w.write(__import__("json").dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            if cfg.max_samples > 0 and n_written >= cfg.max_samples:
                break

    print(f"[TeacherCache] wrote {n_written} rows -> {out_path}")
