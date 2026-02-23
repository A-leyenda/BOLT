#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from bolt.data.robo2vlm import iter_examples_any
from bolt.data.split import split_by_image_id
from bolt.utils.io import ensure_dir, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Prepare (train_kd, val, test) splits by image-id disjointness.")
    p.add_argument("--train_json", type=str, required=True)
    p.add_argument("--test_json", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--val_size", type=int, default=6780)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)

    train_raw = list(iter_examples_any(args.train_json))
    test_raw = list(iter_examples_any(args.test_json))

    res = split_by_image_id(train_raw, test_raw, val_size=args.val_size, seed=args.seed)

    write_jsonl(out_dir / "train_kd.jsonl", res.train_kd)
    write_jsonl(out_dir / "val.jsonl", res.val)
    write_jsonl(out_dir / "test.jsonl", res.test)

    print(f"[Split] train_kd: {len(res.train_kd)}  val: {len(res.val)}  test: {len(res.test)}")
    print(f"[Split] saved to: {out_dir}")


if __name__ == "__main__":
    main()
