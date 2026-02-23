from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogisticGain(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)


@dataclass
class GainModels:
    # action -> model
    models: Dict[str, LogisticGain]
    in_dim: int

    def save(self, path: str) -> None:
        payload = {
            "in_dim": self.in_dim,
            "state": {k: v.state_dict() for k, v in self.models.items()},
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str) -> "GainModels":
        payload = torch.load(path, map_location="cpu")
        in_dim = int(payload["in_dim"])
        models = {}
        for k, sd in payload["state"].items():
            m = LogisticGain(in_dim)
            m.load_state_dict(sd)
            m.eval()
            models[k] = m
        return GainModels(models=models, in_dim=in_dim)

    def predict(self, action: str, feats: np.ndarray) -> float:
        m = self.models[action]
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            p = float(m(x).item())
        return p


def train_logistic(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    epochs: int = 50,
) -> LogisticGain:
    """Train a simple logistic regression in torch."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    model = LogisticGain(in_dim=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    x_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for _ in range(epochs):
        model.train()
        p = model(x_t)
        loss = F.binary_cross_entropy(p, y_t)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    return model
