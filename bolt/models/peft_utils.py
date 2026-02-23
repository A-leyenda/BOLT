from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Tuple

from transformers import AutoProcessor

from bolt.models.loaders import LoadedVLM, load_vlm


def load_student_with_adapter(
    ckpt_dir: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
) -> LoadedVLM:
    """Load a base model + a LoRA adapter checkpoint saved by PEFT.

    If `ckpt_dir` is a full model directory (no adapter_config.json),
    we load it directly as a model.
    """
    ckpt = Path(ckpt_dir)
    adapter_cfg_path = ckpt / "adapter_config.json"

    if adapter_cfg_path.exists():
        adapter_cfg = json.loads(adapter_cfg_path.read_text(encoding="utf-8"))
        base_id = adapter_cfg.get("base_model_name_or_path")
        if not base_id:
            raise ValueError(f"adapter_config.json found but base_model_name_or_path missing: {adapter_cfg_path}")

        loaded = load_vlm(
            base_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            trust_remote_code=trust_remote_code,
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(loaded.model, ckpt_dir)
        processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=trust_remote_code)
        return LoadedVLM(model=model, processor=processor, kind=loaded.kind)

    # full model dir
    loaded = load_vlm(
        ckpt_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        trust_remote_code=trust_remote_code,
    )
    return loaded
