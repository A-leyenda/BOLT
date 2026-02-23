from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def accuracy(pred_idx: Sequence[int], gt_idx: Sequence[int]) -> float:
    pred = np.asarray(pred_idx)
    gt = np.asarray(gt_idx)
    if pred.size == 0:
        return 0.0
    return float(np.mean(pred == gt))


def nll(probs: Sequence[Sequence[float]], gt_idx: Sequence[int], eps: float = 1e-12) -> float:
    ps = np.asarray(probs, dtype=np.float64)
    gt = np.asarray(gt_idx, dtype=np.int64)
    ps = np.clip(ps, eps, 1.0)
    return float(np.mean(-np.log(ps[np.arange(gt.shape[0]), gt])))


def brier(probs: Sequence[Sequence[float]], gt_idx: Sequence[int]) -> float:
    ps = np.asarray(probs, dtype=np.float64)
    gt = np.asarray(gt_idx, dtype=np.int64)
    N, K = ps.shape
    y = np.zeros((N, K), dtype=np.float64)
    y[np.arange(N), gt] = 1.0
    return float(np.mean(np.sum((ps - y) ** 2, axis=1)))


def ece(probs: Sequence[Sequence[float]], gt_idx: Sequence[int], n_bins: int = 15) -> float:
    ps = np.asarray(probs, dtype=np.float64)
    gt = np.asarray(gt_idx, dtype=np.int64)

    conf = ps.max(axis=1)
    pred = ps.argmax(axis=1)
    acc = (pred == gt).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(mask):
            continue
        w = float(np.mean(mask))
        ece_val += w * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece_val)


def aurc(probs: Sequence[Sequence[float]], gt_idx: Sequence[int]) -> float:
    """Area under risk-coverage curve (AURC).

    We sort by confidence descending. For each coverage c, risk is error rate on top-c fraction.
    AURC is average risk over coverage.
    """
    ps = np.asarray(probs, dtype=np.float64)
    gt = np.asarray(gt_idx, dtype=np.int64)
    conf = ps.max(axis=1)
    pred = ps.argmax(axis=1)
    err = (pred != gt).astype(np.float64)

    order = np.argsort(-conf)
    err_sorted = err[order]
    cum_err = np.cumsum(err_sorted)
    N = len(err_sorted)
    risks = []
    for k in range(1, N + 1):
        risks.append(float(cum_err[k - 1] / k))
    return float(np.mean(risks))


@dataclass
class HallucinationProxies:
    ior: float
    noa_misuse: float
    flip: float
    ho_mean_wrong: float
    ocw_07: float
    rcr: Optional[float] = None
    qdc: Optional[float] = None


def hallucination_proxies(
    *,
    options_list: List[List[str]],
    pred_text_pass1: List[str],
    pred_text_final: List[str],
    pmax_final: List[float],
    gt_text: List[str],
    routed_to_rag: Optional[List[bool]] = None,
    rag_retrieved_answers: Optional[List[List[str]]] = None,
    routed_to_qd: Optional[List[bool]] = None,
    qd_round_preds: Optional[List[List[str]]] = None,
) -> HallucinationProxies:
    """Compute paper-style hallucination proxy metrics.

    We follow the definitions in the paper appendix:
    - IOR: invalid option rate (output not in allowed option set)
    - NOA misuse: predict "None of the above" when GT is not "None of the above"
    - Flip: final label differs from pass-1 label
    - HO mean wrong: mean pmax on wrong predictions
    - OCW@0.7: share of wrong predictions with pmax >= 0.7
    - RCR/QDC: augmentation-conditioned contradiction proxies
    """
    N = len(pred_text_final)
    assert len(options_list) == N

    def is_noa(s: str) -> bool:
        return (s or "").strip().lower() == "none of the above"

    # IOR
    invalid = 0
    for i in range(N):
        if pred_text_final[i] not in options_list[i]:
            invalid += 1
    ior = invalid / max(N, 1)

    # NOA misuse: predicted NOA but GT not NOA
    noa_mis = 0
    for i in range(N):
        if is_noa(pred_text_final[i]) and not is_noa(gt_text[i]):
            noa_mis += 1
    noa_misuse = noa_mis / max(N, 1)

    # Flip
    flip = float(np.mean([pred_text_final[i] != pred_text_pass1[i] for i in range(N)]))

    # Wrong set
    wrong_mask = np.array([pred_text_final[i] != gt_text[i] for i in range(N)], dtype=bool)
    if wrong_mask.any():
        ho_mean_wrong = float(np.mean(np.asarray(pmax_final, dtype=np.float64)[wrong_mask]))
        ocw_07 = float(np.mean((np.asarray(pmax_final, dtype=np.float64)[wrong_mask] >= 0.7).astype(np.float64)))
    else:
        ho_mean_wrong = 0.0
        ocw_07 = 0.0

    # RCR proxy: among routed-to-RAG cases, retrieved answers majority disagrees with GT
    rcr = None
    if routed_to_rag is not None and rag_retrieved_answers is not None:
        flags = []
        for i in range(N):
            if not routed_to_rag[i]:
                continue
            ans = rag_retrieved_answers[i] or []
            if not ans:
                continue
            # majority answer
            uniq, cnt = np.unique(np.array(ans, dtype=object), return_counts=True)
            maj = str(uniq[int(np.argmax(cnt))])
            flags.append(maj != gt_text[i])
        rcr = float(np.mean(flags)) if flags else 0.0

    # QDC proxy: among QD cases, any round pred differs from final pred
    qdc = None
    if routed_to_qd is not None and qd_round_preds is not None:
        flags = []
        for i in range(N):
            if not routed_to_qd[i]:
                continue
            rounds = qd_round_preds[i] or []
            if not rounds:
                continue
            flags.append(any(r != pred_text_final[i] for r in rounds))
        qdc = float(np.mean(flags)) if flags else 0.0

    return HallucinationProxies(
        ior=ior,
        noa_misuse=noa_misuse,
        flip=flip,
        ho_mean_wrong=ho_mean_wrong,
        ocw_07=ocw_07,
        rcr=rcr,
        qdc=qdc,
    )
