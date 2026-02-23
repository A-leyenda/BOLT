from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoConfig, AutoProcessor


@dataclass
class LoadedVLM:
    model: Any
    processor: Any
    kind: str  # e.g. "qwen2-vl", "llava", "paligemma"


def _infer_kind(model_id: str, cfg: Any) -> str:
    mid = (model_id or "").lower()
    mtype = getattr(cfg, "model_type", "") or ""
    mtype = mtype.lower()

    if "qwen2" in mid and "vl" in mid:
        return "qwen2-vl"
    if "qwen2_vl" in mtype or ("qwen2" in mtype and "vl" in mtype):
        return "qwen2-vl"

    if "llava" in mid or "llava" in mtype:
        return "llava"

    if "paligemma" in mid or "paligemma" in mtype:
        return "paligemma"

    # fallback
    return "unknown"


def load_vlm(
    model_id: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
) -> LoadedVLM:
    """Load a HF multimodal LM + processor.

    This wrapper centralizes model-type branching so scripts stay clean.

    Args:
        model_id: HF hub id or local path
        device_map: e.g. "auto" or "cuda:0"
        torch_dtype: "auto" | "float16" | "bfloat16" | "float32"
        load_in_4bit: use bitsandbytes 4-bit quantization (QLoRA-style)
        trust_remote_code: required by some VLMs
    """
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    kind = _infer_kind(model_id, cfg)

    if torch_dtype == "auto":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    else:
        dtype = getattr(torch, torch_dtype)

    quant_cfg = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:
            raise RuntimeError("bitsandbytes / BitsAndBytesConfig not available. Install bitsandbytes.") from e
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    if kind == "qwen2-vl":
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quant_cfg,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        return LoadedVLM(model=model, processor=processor, kind=kind)

    if kind == "llava":
        # For LLaVA HF checkpoints
        # e.g. llava-hf/llava-1.5-7b-hf or llava-hf/llava-1.5-13b-hf
        from transformers import LlavaForConditionalGeneration
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quant_cfg,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        return LoadedVLM(model=model, processor=processor, kind=kind)

    if kind == "paligemma":
        from transformers import PaliGemmaForConditionalGeneration
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            torch_dtype=dtype,
            device_map=device_map,
            quantization_config=quant_cfg,
        )
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        return LoadedVLM(model=model, processor=processor, kind=kind)

    # generic fallback
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quant_cfg,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return LoadedVLM(model=model, processor=processor, kind=kind)
