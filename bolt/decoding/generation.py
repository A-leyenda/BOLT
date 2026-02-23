from __future__ import annotations

from typing import Any, Optional

from PIL import Image
import torch
from transformers import GenerationConfig


def _get_tokenizer(processor: Any):
    return getattr(processor, "tokenizer", processor)


def build_chat(processor: Any, prompt_text: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
            return processor.apply_chat_template(messages, add_generation_prompt=True)
        except Exception:
            pass
    return f"USER: <image>\n{prompt_text}\nASSISTANT:"


@torch.no_grad()
def hf_generate(
    model: Any,
    processor: Any,
    img: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 256,
    min_new_tokens: int = 1,
    do_sample: bool = False,
    temperature: float = 0.2,
) -> str:
    chat = build_chat(processor, prompt_text)
    inputs = processor(text=[chat], images=[img], return_tensors="pt", padding=True)

    device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    for k, v in list(inputs.items()):
        if torch.is_tensor(v):
            inputs[k] = v.to(device)

    gen_cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(min_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
    )
    out_ids = model.generate(**inputs, generation_config=gen_cfg)
    tok = _get_tokenizer(processor)
    text = tok.decode(out_ids[0], skip_special_tokens=True)
    return text
