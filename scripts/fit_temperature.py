#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bolt.data.robo2vlm import iter_examples_any, normalize_example
from bolt.decoding.option_scoring import OptionScorer, softmax_np
from bolt.decoding.prompts import format_constrained_prompt
from bolt.models.peft_utils import load_student_with_adapter
from bolt.utils.io import ensure_dir, safe_open_image, write_json
from bolt.utils.text import match_option_index


def parse_args():
    p = argparse.ArgumentParser(description="Fit temperature scaling tau_cal on val split (minimize NLL).")
    p.add_argument("--ckpt_dir", type=str, required=True, help="Student checkpoint dir (LoRA adapter or full model).")
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--out_path", type=str, required=True)

    p.add_argument("--short_edge", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=-1)

    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out_path)
    ensure_dir(out_path.parent)

    loaded = load_student_with_adapter(
        args.ckpt_dir,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    loaded.model.eval()
    scorer = OptionScorer(model=loaded.model, processor=loaded.processor, kind=loaded.kind)

    # collect logits and gt
    all_scores = []
    all_gt = []

    n = 0
    for raw in tqdm(iter_examples_any(args.val_jsonl), desc="Collect val scores"):
        ex = normalize_example(raw)
        if ex is None:
            continue
        opts = ex.get("options") or []
        if not opts:
            continue
        gt_idx = match_option_index(opts, ex.get("answer", ""))
        if gt_idx is None:
            continue

        img_path = Path(args.image_root) / ex["image"]
        if not img_path.exists():
            continue
        img = safe_open_image(img_path)

        prompt = format_constrained_prompt(ex["question"], opts)
        # get raw scores (tau=1, probs not used)
        out = scorer.score_options(img, prompt, opts, short_edge=args.short_edge, tau=1.0)
        all_scores.append(np.array(out.scores, dtype=np.float64))
        all_gt.append(int(gt_idx))

        n += 1
        if args.max_samples > 0 and n >= args.max_samples:
            break

    if not all_scores:
        raise RuntimeError("No valid samples found for temperature fitting.")

    scores = np.stack(all_scores, axis=0)  # [N, K]
    gts = np.array(all_gt, dtype=np.int64)

    def mean_nll(tau: float) -> float:
        tau = float(max(tau, 1e-6))
        nll = 0.0
        for i in range(scores.shape[0]):
            p = softmax_np(scores[i], tau=tau)
            p = np.clip(p, 1e-12, 1.0)
            nll += -math.log(float(p[gts[i]]))
        return nll / scores.shape[0]

    # 1D search
    try:
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(mean_nll, bounds=(0.05, 10.0), method="bounded")
        best_tau = float(res.x)
        best_nll = float(res.fun)
    except Exception:
        # fallback grid
        grid = np.logspace(math.log10(0.05), math.log10(10.0), num=60)
        vals = [mean_nll(t) for t in grid]
        j = int(np.argmin(vals))
        best_tau = float(grid[j])
        best_nll = float(vals[j])

    write_json(out_path, {"tau_cal": best_tau, "val_nll": best_nll, "n_samples": int(scores.shape[0])})
    print(f"[Temp] tau_cal={best_tau:.4f}  NLL={best_nll:.4f}  saved to {out_path}")


if __name__ == "__main__":
    main()
