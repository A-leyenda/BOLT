from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

from bolt.decoding.option_scoring import resize_short_edge


def _build_chat(processor: Any, prompt_text: str, answer_text: Optional[str], add_generation_prompt: bool) -> str:
    if hasattr(processor, "apply_chat_template"):
        try:
            user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
            msgs = [user_msg]
            if answer_text is not None:
                msgs.append({"role": "assistant", "content": answer_text})
                return processor.apply_chat_template(msgs, add_generation_prompt=False)
            return processor.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt)
        except Exception:
            pass
    if answer_text is None:
        return f"USER: <image>\n{prompt_text}\nASSISTANT:"
    return f"USER: <image>\n{prompt_text}\nASSISTANT: {answer_text}"


def _to_device(device: torch.device, inputs: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def score_options_with_grad(
    model: Any,
    processor: Any,
    img: Image.Image,
    prompt_text: str,
    options: List[str],
    short_edge: Optional[int] = None,
    max_length: int = 4096,
) -> torch.Tensor:
    """Return a tensor of shape [K] with answer-segment log-likelihood scores.

    This is differentiable w.r.t. model parameters (used for ODD training).
    """
    if not options:
        raise ValueError("options must be non-empty")

    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    im = resize_short_edge(img, short_edge)

    # prompt-only length (no gradient needed)
    with torch.no_grad():
        chat_p = _build_chat(processor, prompt_text, answer_text=None, add_generation_prompt=True)
        inp_p = processor(text=[chat_p], images=[im], return_tensors="pt", padding=True)
        inp_p = _to_device(device, inp_p)
        prompt_len = int(inp_p["input_ids"].shape[1])

    # full chats (batched over options)
    chats = [_build_chat(processor, prompt_text, answer_text=o, add_generation_prompt=False) for o in options]
    inp = processor(text=chats, images=[im] * len(chats), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inp = _to_device(device, inp)

    outputs = model(**inp)
    logits = outputs.logits
    input_ids = inp["input_ids"]
    attn = inp.get("attention_mask", torch.ones_like(input_ids))

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_attn = attn[:, 1:]

    logprobs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    pos = torch.arange(shift_labels.size(1), device=token_logp.device).unsqueeze(0)
    thr = max(prompt_len - 1, 0)
    mask = (pos >= thr) & (shift_attn == 1)
    scores = (token_logp * mask).sum(dim=1)  # [K]
    return scores
