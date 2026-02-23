#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from bolt.data.robo2vlm import iter_examples_any
from bolt.decoding.option_scoring import OptionScorer
from bolt.inference.pipeline import BoltConfig, BoltPipeline
from bolt.metrics.metrics import accuracy, aurc, brier, ece, hallucination_proxies, nll
from bolt.models.peft_utils import load_student_with_adapter
from bolt.router.gain_model import GainModels
from bolt.utils.io import ensure_dir, write_json, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate BOLT bTTA pipeline on a split (val/test).")
    p.add_argument("--data_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)

    p.add_argument("--student_ckpt", type=str, required=True)
    p.add_argument("--temperature", type=str, default=None, help="Path to temperature.json with tau_cal.")

    p.add_argument("--retrieval_db", type=str, default=None)
    p.add_argument("--router_ckpt", type=str, default=None)

    p.add_argument("--budget", type=float, default=2.0)
    p.add_argument("--router_tau", type=float, default=0.0, help="Trigger threshold tau (sweep to hit budgets).")
    p.add_argument("--router_max_rounds", type=int, default=2)

    p.add_argument("--kr", type=int, default=4)
    p.add_argument("--kd", type=int, default=3)

    p.add_argument("--desc_mode", type=str, default="none", choices=["none", "dataset", "generate"])

    p.add_argument("--out_jsonl", type=str, required=True)
    p.add_argument("--out_metrics", type=str, default=None)

    p.add_argument("--max_samples", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    out_jsonl = Path(args.out_jsonl)
    ensure_dir(out_jsonl.parent)

    # load student
    loaded = load_student_with_adapter(args.student_ckpt, device_map="auto", torch_dtype="auto", load_in_4bit=False)
    loaded.model.eval()
    scorer = OptionScorer(model=loaded.model, processor=loaded.processor, kind=loaded.kind)

    # temperature
    tau_cal = 1.0
    if args.temperature:
        with Path(args.temperature).open("r", encoding="utf-8") as f:
            tau_cal = float(json.load(f).get("tau_cal", 1.0))

    # router
    router = None
    enable_router = False
    if args.router_ckpt:
        router = GainModels.load(args.router_ckpt)
        enable_router = True

    cfg = BoltConfig(
        kr=args.kr,
        kd=args.kd,
        retrieval_db=args.retrieval_db,
        budget=args.budget,
        enable_router=enable_router,
        router_tau=args.router_tau,
        router_max_rounds=args.router_max_rounds,
        tau_cal=tau_cal,
        desc_mode=args.desc_mode,
    )
    pipe = BoltPipeline(scorer=scorer, image_root=args.image_root, cfg=cfg, router=router)

    gt_idx = []
    pred_idx = []
    probs = []

    # hallucination proxies inputs
    options_list = []
    pred_pass1 = []
    pred_final = []
    pmax_final = []
    gt_texts = []
    routed_to_rag = []
    rag_answers = []
    routed_to_qd = []
    qd_round_preds = []

    rows = []

    n = 0
    for raw in tqdm(iter_examples_any(args.data_jsonl), desc="Evaluate"):
        r = pipe.predict_one(raw)
        if r.get("skip"):
            continue
        rows.append(r)

        opts = r["options"]
        options_list.append(opts)
        pred_pass1.append(r["pass1_pred"])
        pred_final.append(r["final_pred"])
        pmax_final.append(float(r["final_pmax"]))
        gt_texts.append(r.get("gt", ""))

        # indices for metrics
        if r.get("gt_idx") is not None and r["final_pred"] in opts:
            gt_idx.append(int(r["gt_idx"]))
            pred_idx.append(int(opts.index(r["final_pred"])))
            probs.append(r["probs_final"])

        routed_to_rag.append(bool(r.get("routed_to_rag", False)))
        rag_answers.append(r.get("rag_retrieved_answers", []))
        routed_to_qd.append(bool(r.get("routed_to_qd", False)))
        qd_round_preds.append(r.get("qd_round_preds", []))

        n += 1
        if args.max_samples > 0 and n >= args.max_samples:
            break

    # write predictions
    write_jsonl(out_jsonl, rows)
    print(f"[Saved] predictions -> {out_jsonl} (N={len(rows)})")

    # compute metrics (only on samples with gt_idx)
    m = {}
    if gt_idx:
        m["accuracy"] = accuracy(pred_idx, gt_idx)
        m["nll"] = nll(probs, gt_idx)
        m["brier"] = brier(probs, gt_idx)
        m["ece_15"] = ece(probs, gt_idx, n_bins=15)
        m["aurc"] = aurc(probs, gt_idx)

    hall = hallucination_proxies(
        options_list=options_list,
        pred_text_pass1=pred_pass1,
        pred_text_final=pred_final,
        pmax_final=pmax_final,
        gt_text=gt_texts,
        routed_to_rag=routed_to_rag,
        rag_retrieved_answers=rag_answers,
        routed_to_qd=routed_to_qd,
        qd_round_preds=qd_round_preds,
    )
    m["IOR"] = hall.ior
    m["NOA_misuse"] = hall.noa_misuse
    m["Flip"] = hall.flip
    m["HO_mean_wrong"] = hall.ho_mean_wrong
    m["OCW@0.7"] = hall.ocw_07
    if hall.rcr is not None:
        m["RCR"] = hall.rcr
    if hall.qdc is not None:
        m["QDC"] = hall.qdc

    if args.out_metrics:
        out_m = Path(args.out_metrics)
        ensure_dir(out_m.parent)
        write_json(out_m, m)
        print(f"[Saved] metrics -> {out_m}")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
