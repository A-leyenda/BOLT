#!/usr/bin/env python
from __future__ import annotations

import argparse

from bolt.distill.teacher_cache import TeacherCacheConfig, build_teacher_cache
from bolt.models.loaders import load_vlm
from bolt.decoding.option_scoring import OptionScorer
from bolt.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Build teacher option-distribution cache (KD JSONL).")
    p.add_argument("--data_jsonl", type=str, required=True, help="Train-kd JSON/JSONL.")
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--teacher_id", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)

    p.add_argument("--tau_kd", type=float, default=2.0)
    p.add_argument("--short_edge", type=int, default=512)
    p.add_argument("--max_samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    loaded = load_vlm(
        args.teacher_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=False,
        trust_remote_code=True,
    )
    model, processor, kind = loaded.model, loaded.processor, loaded.kind
    model.eval()

    scorer = OptionScorer(model=model, processor=processor, kind=kind)

    cfg = TeacherCacheConfig(
        data_path=args.data_jsonl,
        image_root=args.image_root,
        out_jsonl=args.out_jsonl,
        tau_kd=args.tau_kd,
        short_edge=args.short_edge,
        max_samples=args.max_samples,
    )
    build_teacher_cache(scorer, cfg)


if __name__ == "__main__":
    main()
