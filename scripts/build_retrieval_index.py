#!/usr/bin/env python
from __future__ import annotations

import argparse

from bolt.retrieval.build import BuildIndexConfig, build_retrieval_index


def parse_args():
    p = argparse.ArgumentParser(description="Build tmRAG retrieval DB from train-kd split.")
    p.add_argument("--data_jsonl", type=str, required=True)
    p.add_argument("--image_root", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--desc_mode", type=str, default="none", choices=["none", "dataset", "generate"])
    p.add_argument("--student_ckpt", type=str, default=None, help="Required if desc_mode=generate.")

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_samples", type=int, default=-1)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = BuildIndexConfig(
        data_jsonl=args.data_jsonl,
        image_root=args.image_root,
        out_dir=args.out_dir,
        embed_model=args.embed_model,
        desc_mode=args.desc_mode,
        student_ckpt=args.student_ckpt,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
    build_retrieval_index(cfg)


if __name__ == "__main__":
    main()
