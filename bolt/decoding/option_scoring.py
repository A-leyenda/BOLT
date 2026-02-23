from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch


def resize_short_edge(img: Image.Image, shortest: Optional[int]) -> Image.Image:
    if not shortest:
        return img
    w, h = img.size
    if min(w, h) == shortest:
        return img
    if w < h:
        new_w = shortest
        new_h = int(round(h * shortest / w))
    else:
        new_h = shortest
        new_w = int(round(w * shortest / h))
    return img.resize((new_w, new_h), Image.LANCZOS)


def softmax_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64) / max(float(tau), 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


@dataclass
class OptionScores:
    scores: List[float]
    probs: List[float]
    pred: str
    pred_idx: int


class OptionScorer:
    """Constrained decoding by scoring answer-segment likelihoods for each option.

    This implements the core interface used by ODD/bTTA:
    - score option k by sum_{t in answer segment} log p(a_t^(k) | prefix)
    - p(option=k) = softmax(score_k / tau)

    Notes
    -----
    - We compute the prompt length L0 from a *prompt-only* chat, and then score
      tokens after L0 in a *prompt+answer* chat.
    - This avoids template misalignment bugs (prompt tokens vs answer tokens).
    """

    def __init__(self, model: Any, processor: Any, kind: str, device: Optional[str] = None):
        self.model = model
        self.processor = processor
        self.kind = kind
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _build_chat(self, prompt_text: str, answer_text: Optional[str], add_generation_prompt: bool) -> str:
        # Prefer processor.apply_chat_template when available (Qwen2-VL / LLaVA HF)
        if hasattr(self.processor, "apply_chat_template"):
            try:
                user_msg = {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
                msgs = [user_msg]
                if answer_text is not None:
                    msgs.append({"role": "assistant", "content": answer_text})
                    return self.processor.apply_chat_template(msgs, add_generation_prompt=False)
                return self.processor.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt)
            except Exception:
                pass

        # Fallback: classic LLaVA-style
        if answer_text is None:
            return f"USER: <image>\n{prompt_text}\nASSISTANT:"
        return f"USER: <image>\n{prompt_text}\nASSISTANT: {answer_text}"

    def _to_device(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in inputs.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    @torch.no_grad()
    def score_options(
        self,
        img: Image.Image,
        prompt_text: str,
        options: List[str],
        short_edge: Optional[int] = None,
        tau: float = 1.0,
        max_length: int = 4096,
    ) -> OptionScores:
        """Return option scores + probabilities + argmax prediction."""
        if not options:
            raise ValueError("options must be a non-empty list of strings")

        im = resize_short_edge(img, short_edge)

        # prompt-only to get L0
        chat_p = self._build_chat(prompt_text, answer_text=None, add_generation_prompt=True)
        inp_p = self.processor(text=[chat_p], images=[im], return_tensors="pt", padding=True)
        inp_p = self._to_device(inp_p)
        prompt_len = int(inp_p["input_ids"].shape[1])

        # prompt+answer batch
        chats = [self._build_chat(prompt_text, answer_text=o, add_generation_prompt=False) for o in options]
        inp = self.processor(text=chats, images=[im] * len(chats), return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        inp = self._to_device(inp)

        outputs = self.model(**inp)
        logits = outputs.logits  # [K, L, V]
        input_ids = inp["input_ids"]  # [K, L]
        attn = inp.get("attention_mask", torch.ones_like(input_ids))

        # shift for causal scoring
        shift_logits = logits[:, :-1, :]  # [K, L-1, V]
        shift_labels = input_ids[:, 1:]   # [K, L-1]
        shift_attn = attn[:, 1:]          # [K, L-1]

        logprobs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)  # [K, L-1]

        # mask positions corresponding to answer segment
        # original token positions i >= prompt_len  <=> shifted index t=i-1 >= prompt_len-1
        pos = torch.arange(shift_labels.size(1), device=token_logp.device).unsqueeze(0)  # [1, L-1]
        thr = max(prompt_len - 1, 0)
        mask = (pos >= thr) & (shift_attn == 1)

        # sum logp over answer tokens
        seq_scores = (token_logp * mask).sum(dim=1)  # [K]

        scores = seq_scores.detach().float().cpu().tolist()
        probs = softmax_np(np.array(scores, dtype=np.float64), tau=tau).tolist()
        pred_idx = int(np.argmax(probs))
        pred = options[pred_idx]
        return OptionScores(scores=scores, probs=probs, pred=pred, pred_idx=pred_idx)
