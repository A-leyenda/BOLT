from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class RoutingFeatures:
    pmax: float
    margin: float
    entropy: float
    rho: float = 0.0
    kappa: float = 0.0

    def as_vector(self, include_kappa: bool = True) -> np.ndarray:
        if include_kappa:
            return np.array([self.pmax, self.margin, self.entropy, self.rho, self.kappa], dtype=np.float32)
        return np.array([self.pmax, self.margin, self.entropy, self.rho], dtype=np.float32)


def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    return float(-(p * np.log(p)).sum())


def basic_features(probs: List[float]) -> Tuple[float, float, float]:
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0.0, 0.0, 0.0
    p_sorted = np.sort(p)[::-1]
    pmax = float(p_sorted[0])
    p2 = float(p_sorted[1]) if p_sorted.size > 1 else 0.0
    margin = pmax - p2
    ent = _entropy(p)
    return pmax, margin, ent


def retrieval_affinity(top_sims: List[float]) -> float:
    if not top_sims:
        return 0.0
    return float(np.mean(top_sims))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
    kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
    return 0.5 * (kl_pm + kl_qm)


def agreement_kappa(dists: List[List[float]]) -> float:
    """κ = 1 - average pairwise JS divergence over distributions."""
    if not dists or len(dists) < 2:
        return 1.0
    ps = [np.asarray(p, dtype=np.float64) for p in dists]
    Kd = len(ps)
    s = 0.0
    cnt = 0
    for i in range(Kd):
        for j in range(i + 1, Kd):
            s += js_divergence(ps[i], ps[j])
            cnt += 1
    avg = s / max(cnt, 1)
    kappa = 1.0 - avg
    # clamp to [0,1]
    return float(max(0.0, min(1.0, kappa)))
