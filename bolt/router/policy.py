from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from bolt.router.gain_model import GainModels
from bolt.router.features import RoutingFeatures

# Paper default normalized marginal costs (Appendix F / Sec. 4.4)
DEFAULT_COSTS = {"HR": 0.50, "tmRAG": 0.30, "QD": 0.35}


@dataclass
class PolicyConfig:
    """Budgeted routing policy (Algorithm in Appendix B).

    - Base pass costs 1.00.
    - Each action a has marginal cost C_a.
    - Router provides gω(f,a) ≈ Pr[ΔAcc_a=1|f].
    - Decision score: u_a = gω(f,a) * W_a / C_a.
    - Greedy for at most `max_rounds` rounds:
        if max_a u_a < tau: stop; else apply best action.
    """

    budget: float = 2.00
    base_cost: float = 1.00
    costs: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_COSTS))
    weights: Dict[str, float] = field(default_factory=lambda: {"HR": 1.0, "tmRAG": 1.0, "QD": 1.0})
    max_rounds: int = 2
    tau: float = 0.0  # trigger threshold (sweep this to trace budget-accuracy frontier)


def choose_next_action(
    gain_models: GainModels,
    feats: RoutingFeatures,
    cfg: PolicyConfig,
    remaining_budget: float,
    used_actions: List[str],
) -> Optional[str]:
    """Return the next action to execute, or None (stop)."""
    vec = feats.as_vector(include_kappa=False)  # [pmax, margin, entropy, rho]
    best_a: Optional[str] = None
    best_u = -1e9
    for a, cost in cfg.costs.items():
        if a in used_actions:
            continue
        if cost > remaining_budget + 1e-9:
            continue
        p = gain_models.predict(a, vec)
        u = (p * float(cfg.weights.get(a, 1.0))) / max(float(cost), 1e-6)
        if u > best_u:
            best_u = u
            best_a = a
    if best_a is None:
        return None
    if best_u < cfg.tau:
        return None
    return best_a
