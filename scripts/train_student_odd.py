#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
from tqdm import tqdm

from bolt.decoding.prompts import format_constrained_prompt
from bolt.distill.kd_dataset import iter_kd_samples
from bolt.distill.odd_loss import ODDLossConfig, odd_loss
from bolt.distill.student_scoring import score_options_with_grad
from bolt.models.loaders import load_vlm
from bolt.utils.io import ensure_dir, safe_open_image
from bolt.utils.seed import set_seed


def parse_args():
    p = argparse.ArgumentParser(description="Train a student VLM with Option-level Decision Distillation (ODD).")

    p.add_argument("--kd_jsonl", type=str, required=True, help="Teacher cache JSONL with teacher_probs.")
    p.add_argument("--image_root", type=str, required=True, help="Root dir containing images.")

    p.add_argument("--student_id", type=str, required=True, help="HF id or local path of the student base model.")
    p.add_argument("--output_dir", type=str, required=True, help="Where to save LoRA adapter (and processor).")

    # training
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=1, help="Per-step batch size (recommended 1 for VLMs).")
    p.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps.")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_steps", type=int, default=-1, help="Stop after N optimizer steps (<=0 means full epochs).")

    # ODD loss weights
    p.add_argument("--lambda_kl", type=float, default=1.0)
    p.add_argument("--lambda_ce", type=float, default=0.1)
    p.add_argument("--label_smoothing", type=float, default=0.0)

    # model loading
    p.add_argument("--load_in_4bit", action="store_true", help="Enable 4-bit quantization (QLoRA).")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--torch_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])

    # lora
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--short_edge", type=int, default=512, help="Resize shortest edge before scoring.")
    p.add_argument("--log_every", type=int, default=50)

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = ensure_dir(args.output_dir)

    # 1) load student base
    loaded = load_vlm(
        args.student_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=True,
    )
    model, processor, kind = loaded.model, loaded.processor, loaded.kind
    model.train()

    # 2) attach LoRA
    from peft import LoraConfig, get_peft_model, TaskType

    target_modules = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 3) optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    loss_cfg = ODDLossConfig(
        lambda_kl=args.lambda_kl,
        lambda_ce=args.lambda_ce,
        label_smoothing=args.label_smoothing,
    )

    # 4) training loop (streaming)
    kd_path = Path(args.kd_jsonl)
    img_root = Path(args.image_root)

    # We keep it simple: batch_size=1 by default.
    # If you set batch_size>1, we process examples sequentially inside the batch
    # (still correct but slower). For speed you can implement your own batching strategy.
    global_step = 0
    opt_step = 0
    running = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0}

    def log(prefix: str):
        n = max(running["n"], 1)
        print(
            f"{prefix} step={opt_step} "
            f"loss={running['loss']/n:.4f} kl={running['kl']/n:.4f} ce={running['ce']/n:.4f}"
        )

    for epoch in range(args.epochs):
        pbar = tqdm(iter_kd_samples(kd_path), desc=f"ODD epoch {epoch+1}/{args.epochs}")
        for sample in pbar:
            model.train()
            img_path = img_root / sample.image
            if not img_path.exists():
                continue
            img = safe_open_image(img_path)

            prompt = format_constrained_prompt(sample.question, sample.options)
            scores = score_options_with_grad(
                model=model,
                processor=processor,
                img=img,
                prompt_text=prompt,
                options=sample.options,
                short_edge=args.short_edge,
            )
            t_probs = torch.tensor(sample.teacher_probs, dtype=torch.float32, device=scores.device)
            loss, kl, ce = odd_loss(scores, t_probs, sample.gt_idx, loss_cfg)

            loss = loss / max(1, args.grad_accum)
            loss.backward()

            running["loss"] += float(loss.detach().cpu().item()) * max(1, args.grad_accum)
            running["kl"] += float(kl.detach().cpu().item())
            running["ce"] += float(ce.detach().cpu().item())
            running["n"] += 1

            global_step += 1
            if global_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1

                if args.log_every > 0 and opt_step % args.log_every == 0:
                    log("[Train]")
                    running = {"loss": 0.0, "kl": 0.0, "ce": 0.0, "n": 0}

                if args.max_steps > 0 and opt_step >= args.max_steps:
                    break

        if args.max_steps > 0 and opt_step >= args.max_steps:
            break

    # Save adapter + processor
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"[Done] Saved LoRA adapter + processor to: {out_dir}")


if __name__ == "__main__":
    main()
