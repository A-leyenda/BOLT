from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class ODDLossConfig:
    lambda_kl: float = 1.0
    lambda_ce: float = 0.1
    label_smoothing: float = 0.0  # 0 = one-hot
    eps: float = 1e-8


def odd_loss(
    student_scores: torch.Tensor,  # [K]
    teacher_probs: torch.Tensor,   # [K]
    gt_idx: Optional[int],
    cfg: ODDLossConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute ODD loss for a single example.

    p_S = softmax(student_scores)
    KL(p_T || p_S) + CE(δ_y || p_S)

    Returns:
        total, kl, ce
    """
    # student distribution
    p_s = torch.softmax(student_scores, dim=-1).clamp(min=cfg.eps, max=1.0)
    log_p_s = torch.log(p_s)

    p_t = teacher_probs / (teacher_probs.sum() + cfg.eps)
    p_t = p_t.clamp(min=cfg.eps, max=1.0)
    log_p_t = torch.log(p_t)

    kl = torch.sum(p_t * (log_p_t - log_p_s))

    ce = torch.tensor(0.0, device=student_scores.device)
    if gt_idx is not None and 0 <= int(gt_idx) < int(student_scores.numel()):
        K = int(student_scores.numel())
        if cfg.label_smoothing > 0:
            # y_smooth = (1-ε) δ_y + ε/K
            y = torch.full((K,), cfg.label_smoothing / K, device=student_scores.device)
            y[int(gt_idx)] += 1.0 - cfg.label_smoothing
            ce = -torch.sum(y * log_p_s)
        else:
            ce = -log_p_s[int(gt_idx)]

    total = cfg.lambda_kl * kl + cfg.lambda_ce * ce
    return total, kl.detach(), ce.detach()
