#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from bolt.data.robo2vlm import iter_examples_any, normalize_example
from bolt.decoding.option_scoring import OptionScorer
from bolt.inference.pipeline import BoltConfig, BoltPipeline
from bolt.models.peft_utils import load_student_with_adapter
from bolt.router.features import basic_features, retrieval_affinity, RoutingFeatures
from bolt.router.gain_model import GainModels, train_logistic
from bolt.utils.io import ensure_dir, safe_open_image, write_json
from bolt.utils.text import match_option_index


def parse_args():
    p = argparse.ArgumentParser(description="Train bTTA router gain models on val split (offline).")
    p.add_argument("--val_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)

    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--retrieval_db", type=str, default=None)

    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--kr", type=int, default=4)
    p.add_argument("--kd", type=int, default=3)
    p.add_argument("--short_edge_base", type=int, default=512)
    p.add_argument("--short_edge_hr", type=int, default=1024)

    p.add_argument("--max_samples", type=int, default=-1)

    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-2)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    router_path = out_dir / "router.pt"
    info_path = out_dir / "router_info.json"

    loaded = load_student_with_adapter(args.student_ckpt, device_map="auto", torch_dtype="auto", load_in_4bit=False)
    loaded.model.eval()
    scorer = OptionScorer(model=loaded.model, processor=loaded.processor, kind=loaded.kind)

    cfg = BoltConfig(
        kr=args.kr,
        kd=args.kd,
        retrieval_db=args.retrieval_db,
        short_edge_base=args.short_edge_base,
        short_edge_hr=args.short_edge_hr,
        enable_router=False,  # training stage
        tau_cal=1.0,
        budget=args.budget,
        desc_mode="none",
    )
    pipe = BoltPipeline(scorer=scorer, image_root=args.image_root, cfg=cfg, router=None)

    X = []
    y_hr = []
    y_rag = []
    y_qd = []

    n = 0
    for raw in tqdm(iter_examples_any(args.val_jsonl), desc="Collect router logs"):
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

        # base pass1
        out1, probs1_cal, _ = pipe._score(img, ex["question"], opts, short_edge=args.short_edge_base, extra_prefix="")
        c1 = int(out1.pred == opts[int(gt_idx)])

        # retrieval affinity (cheap)
        rho, retrieved_items, retrieved_block = pipe._retrieval(ex["question"], opts, desc=None, image_name=ex["image"])

        pmax, margin, ent = basic_features(probs1_cal)
        feats = RoutingFeatures(pmax=pmax, margin=margin, entropy=ent, rho=rho, kappa=0.0)
        X.append(feats.as_vector(include_kappa=False))  # [4]

        # HR
        out_hr, probs_hr, _ = pipe._score(img, ex["question"], opts, short_edge=args.short_edge_hr, extra_prefix="")
        c_hr = int(out_hr.pred == opts[int(gt_idx)])
        y_hr.append(1 if c_hr > c1 else 0)

        # tmRAG
        extra = "[Retrieved examples]\n" + retrieved_block if retrieved_block else ""
        out_rag, probs_rag, _ = pipe._score(img, ex["question"], opts, short_edge=args.short_edge_base, extra_prefix=extra)
        c_rag = int(out_rag.pred == opts[int(gt_idx)])
        y_rag.append(1 if c_rag > c1 else 0)

        # QD (can be expensive)
        tkey = ex.get("type") or "other"
        p_hat, kappa, round_preds = pipe._qd(img, ex["question"], opts, desc=None, retrieved_block=retrieved_block, type_key=tkey)
        pred_qd = opts[int(np.argmax(np.asarray(p_hat)))]
        c_qd = int(pred_qd == opts[int(gt_idx)])
        y_qd.append(1 if c_qd > c1 else 0)

        n += 1
        if args.max_samples > 0 and n >= args.max_samples:
            break

    X = np.asarray(X, dtype=np.float32)
    y_hr = np.asarray(y_hr, dtype=np.float32)
    y_rag = np.asarray(y_rag, dtype=np.float32)
    y_qd = np.asarray(y_qd, dtype=np.float32)

    print(f"[Router] Collected N={X.shape[0]} samples. Train logistic models...")

    m_hr = train_logistic(X, y_hr, lr=args.lr, epochs=args.epochs)
    m_rag = train_logistic(X, y_rag, lr=args.lr, epochs=args.epochs)
    m_qd = train_logistic(X, y_qd, lr=args.lr, epochs=args.epochs)

    models = GainModels(models={"HR": m_hr, "tmRAG": m_rag, "QD": m_qd}, in_dim=X.shape[1])
    models.save(str(router_path))

    info = {
        "features": ["pmax", "margin", "entropy", "rho"],
        "N": int(X.shape[0]),
        "positive_rate": {
            "HR": float(y_hr.mean()) if len(y_hr) else 0.0,
            "tmRAG": float(y_rag.mean()) if len(y_rag) else 0.0,
            "QD": float(y_qd.mean()) if len(y_qd) else 0.0,
        },
        "costs": {"HR": 0.50, "tmRAG": 0.30, "QD": 0.35},
    }
    write_json(info_path, info)
    print(f"[Saved] router -> {router_path}")
    print(f"[Saved] router_info -> {info_path}")


if __name__ == "__main__":
    main()
