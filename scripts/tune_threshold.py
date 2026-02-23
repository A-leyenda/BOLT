#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bolt.data.robo2vlm import iter_examples_any
from bolt.decoding.option_scoring import OptionScorer
from bolt.inference.pipeline import BoltConfig, BoltPipeline
from bolt.models.peft_utils import load_student_with_adapter
from bolt.router.gain_model import GainModels
from bolt.utils.io import ensure_dir, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Tune router threshold tau to hit a target average budget on val.")
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)

    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--router_ckpt", type=str, required=True)
    p.add_argument("--retrieval_db", type=str, default=None)

    p.add_argument("--target_budget", type=float, default=2.0)
    p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--max_iter", type=int, default=12)
    p.add_argument("--tau_low", type=float, default=0.0)
    p.add_argument("--tau_high", type=float, default=5.0)

    p.add_argument("--kr", type=int, default=4)
    p.add_argument("--kd", type=int, default=3)

    p.add_argument("--out_path", type=str, required=True)
    p.add_argument("--max_samples", type=int, default=-1)
    return p.parse_args()


def eval_tau(pipe: BoltPipeline, data_jsonl: str, max_samples: int) -> tuple[float, float]:
    budgets = []
    correct = []
    n = 0
    for raw in tqdm(iter_examples_any(data_jsonl), desc=f"tau={pipe.cfg.router_tau:.3f}", leave=False):
        r = pipe.predict_one(raw)
        if r.get("skip"):
            continue
        budgets.append(float(r["used_budget"]))
        gt_idx = r.get("gt_idx")
        if gt_idx is not None:
            correct.append(1.0 if r["final_pred"] == r["options"][int(gt_idx)] else 0.0)
        n += 1
        if max_samples > 0 and n >= max_samples:
            break
    avg_budget = float(np.mean(budgets)) if budgets else 0.0
    acc = float(np.mean(correct)) if correct else 0.0
    return avg_budget, acc


def main():
    args = parse_args()
    out_path = Path(args.out_path)
    ensure_dir(out_path.parent)

    # model
    loaded = load_student_with_adapter(args.student_ckpt, device_map="auto", torch_dtype="auto", load_in_4bit=False)
    loaded.model.eval()
    scorer = OptionScorer(model=loaded.model, processor=loaded.processor, kind=loaded.kind)

    router = GainModels.load(args.router_ckpt)

    low, high = float(args.tau_low), float(args.tau_high)
    best = None

    for it in range(args.max_iter):
        mid = 0.5 * (low + high)
        cfg = BoltConfig(
            kr=args.kr,
            kd=args.kd,
            retrieval_db=args.retrieval_db,
            enable_router=True,
            budget=args.target_budget,
            router_tau=mid,
        )
        pipe = BoltPipeline(scorer=scorer, image_root=args.image_root, cfg=cfg, router=router)

        avg_budget, acc = eval_tau(pipe, args.val_jsonl, args.max_samples)
        rec = {"iter": it, "tau": mid, "avg_budget": avg_budget, "acc": acc}
        print(json.dumps(rec))

        best = rec if best is None else min([best, rec], key=lambda r: abs(r["avg_budget"] - args.target_budget))

        if abs(avg_budget - args.target_budget) <= args.tol:
            break
        if avg_budget > args.target_budget:
            # too expensive -> increase tau (trigger less)
            low = mid
        else:
            high = mid

    write_json(out_path, {"best": best, "tau_low": low, "tau_high": high})
    print(f"[Saved] {out_path}")
    print(f"[Best] {best}")


if __name__ == "__main__":
    main()
