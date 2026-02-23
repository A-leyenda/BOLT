from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from bolt.decoding.generation import hf_generate
from bolt.decoding.option_scoring import OptionScorer, softmax_np
from bolt.decoding.prompts import (
    DEFAULT_DESC_PROMPT,
    DEFAULT_QD_GUIDELINE,
    format_constrained_prompt,
    format_qd_preamble,
)
from bolt.retrieval.db import RetrievalDB, RetrievalItem
from bolt.router.features import RoutingFeatures, agreement_kappa, basic_features, retrieval_affinity
from bolt.router.gain_model import GainModels
from bolt.router.policy import PolicyConfig, choose_next_action
from bolt.utils.io import safe_open_image
from bolt.utils.text import match_option_index, question_type_key


class QueryEmbedder:
    def __init__(self, embed_model: str):
        from sentence_transformers import SentenceTransformer
        try:
            self.model = SentenceTransformer(embed_model, device="cpu")
        except Exception:
            self.model = SentenceTransformer(embed_model)

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype=np.float32)


def _make_text_for_embed(question: str, desc: Optional[Dict[str, str]]) -> str:
    d_en = (desc or {}).get("en", "")
    d_zh = (desc or {}).get("zh", "")
    d_en = d_en[:256]
    d_zh = d_zh[:256]
    return f"Q: {question.strip()}\nEN: {d_en}\nZH: {d_zh}".strip()


def _build_retrieved_examples_block(items: List[RetrievalItem]) -> str:
    blocks = []
    for it in items:
        d = it.desc or {}
        en = (d.get("en", "") or "")[:220]
        zh = (d.get("zh", "") or "")[:220]
        blocks.append(
            "Example\n"
            f"EN: {en}\n"
            f"ZH: {zh}\n"
            f"Q: {it.question}\n"
            f"A: {it.answer}\n"
        )
    return "\n".join(blocks).strip()


@dataclass
class BoltConfig:
    # base pass
    short_edge_base: int = 512
    short_edge_hr: int = 1024

    # retrieval
    kr: int = 4
    retrieval_db: Optional[str] = None
    embed_model: Optional[str] = None  # if None, read from retrieval_db/embed_model.json

    # QD
    kd: int = 3
    qd_gen_tokens: int = 128
    qd_sample: bool = True
    qd_temperature: float = 0.7

    # description
    desc_mode: str = "none"  # none|generate|dataset
    desc_prompt: str = DEFAULT_DESC_PROMPT

    # routing
    budget: float = 2.0
    enable_router: bool = True
    router_tau: float = 0.0
    router_max_rounds: int = 2

    # probability calibration
    tau_cal: float = 1.0


class BoltPipeline:
    """End-to-end BOLT bTTA pipeline.

    This is intentionally modular:
    - pass-1 constrained decoding via OptionScorer (answer-segment likelihood)
    - optional tmRAG retrieval prefix
    - optional QD (generate short decompositions, then score options)
    - budgeted routing controlled by a learned gain model + threshold
    """

    def __init__(
        self,
        scorer: OptionScorer,
        image_root: str,
        cfg: BoltConfig,
        router: Optional[GainModels] = None,
    ):
        self.scorer = scorer
        self.image_root = Path(image_root)
        self.cfg = cfg
        self.router = router

        self.db: Optional[RetrievalDB] = None
        self.embedder: Optional[QueryEmbedder] = None
        self.embed_model: Optional[str] = None

        if cfg.retrieval_db:
            self.db = RetrievalDB(cfg.retrieval_db)
            em = cfg.embed_model
            if em is None:
                info_path = Path(cfg.retrieval_db) / "embed_model.json"
                if info_path.exists():
                    try:
                        em = json.loads(info_path.read_text(encoding="utf-8")).get("embed_model")
                    except Exception:
                        em = None
            self.embed_model = em or "sentence-transformers/all-MiniLM-L6-v2"
            self.embedder = QueryEmbedder(self.embed_model)

    def _get_desc(self, raw_ex: Dict[str, Any], img: Image.Image) -> Optional[Dict[str, str]]:
        if self.cfg.desc_mode == "dataset":
            d = raw_ex.get("desc") or raw_ex.get("description") or raw_ex.get("caption")
            if isinstance(d, dict):
                return {"en": str(d.get("en", "") or ""), "zh": str(d.get("zh", "") or "")}
            if isinstance(d, str) and d.strip():
                return {"en": d.strip(), "zh": ""}
            return None

        if self.cfg.desc_mode == "generate":
            txt = hf_generate(
                self.scorer.model,
                self.scorer.processor,
                img,
                self.cfg.desc_prompt,
                max_new_tokens=256,
                min_new_tokens=16,
                do_sample=False,
                temperature=0.2,
            )
            return {"en": txt, "zh": ""}
        return None

    def _score(self, img: Image.Image, question: str, options: List[str], short_edge: int, extra_prefix: str = ""):
        prompt = format_constrained_prompt(question, options)
        if extra_prefix:
            prompt = extra_prefix.strip() + "\n\n" + prompt
        out = self.scorer.score_options(img, prompt, options, short_edge=short_edge, tau=1.0)
        probs_cal = softmax_np(np.array(out.scores, dtype=np.float64), tau=self.cfg.tau_cal).tolist()
        return out, probs_cal, prompt

    def _retrieval(self, question: str, options: List[str], desc: Optional[Dict[str, str]], image_name: str) -> Tuple[float, List[RetrievalItem], str]:
        if not self.db or not self.embedder:
            return 0.0, [], ""
        tkey = question_type_key(question, options)
        qtext = _make_text_for_embed(question, desc)
        qvec = self.embedder.encode(qtext)
        retrieved = self.db.retrieve(qvec, type_key=tkey, topk=self.cfg.kr, exclude_image=image_name)
        sims = [s for s, _ in retrieved]
        items = [it for _, it in retrieved]
        rho = retrieval_affinity(sims)
        block = _build_retrieved_examples_block(items)
        return rho, items, block

    def _qd(self, img: Image.Image, question: str, options: List[str], desc: Optional[Dict[str, str]], retrieved_block: str, type_key: str) -> Tuple[List[float], float, List[str]]:
        dists = []
        preds = []
        for _ in range(self.cfg.kd):
            pre = format_qd_preamble(
                type_key=type_key,
                question=question,
                options=options,
                desc=desc,
                retrieved_examples_block=retrieved_block,
                guideline=DEFAULT_QD_GUIDELINE,
            )
            gen_prompt = pre + "\n\nWrite ONLY the 2-4 verification sub-questions (one per line)."
            decomp = hf_generate(
                self.scorer.model,
                self.scorer.processor,
                img,
                gen_prompt,
                max_new_tokens=self.cfg.qd_gen_tokens,
                min_new_tokens=16,
                do_sample=self.cfg.qd_sample,
                temperature=self.cfg.qd_temperature,
            )
            # keep only the last few non-empty lines
            decomp_lines = [ln.strip() for ln in decomp.splitlines() if ln.strip()]
            decomp_short = "\n".join(decomp_lines[-8:])

            extra = pre + "\n\n[Verification questions]\n" + decomp_short
            out, probs_cal, _ = self._score(img, question, options, short_edge=self.cfg.short_edge_base, extra_prefix=extra)
            dists.append(probs_cal)
            preds.append(out.pred)

        p_hat = np.mean(np.asarray(dists, dtype=np.float64), axis=0)
        p_hat = (p_hat / (p_hat.sum() + 1e-12)).tolist()
        kappa = agreement_kappa(dists)
        return p_hat, kappa, preds

    def predict_one(self, raw_ex: Dict[str, Any]) -> Dict[str, Any]:
        from bolt.data.robo2vlm import normalize_example

        ex = normalize_example(raw_ex)
        if ex is None:
            return {"skip": True}

        img_path = self.image_root / ex["image"]
        if not img_path.exists():
            return {"skip": True, "reason": "missing_image", "image": ex["image"]}

        img = safe_open_image(img_path)
        options = ex.get("options") or []
        if not options:
            return {"skip": True, "reason": "missing_options", "image": ex["image"]}

        gt_idx = match_option_index(options, ex.get("answer", ""))
        gt_text = ex.get("answer", "")

        # description (optional)
        desc = self._get_desc(raw_ex, img)

        # precompute retrieval block + rho (cheap)
        rho, retrieved_items, retrieved_block = self._retrieval(ex["question"], options, desc, ex["image"])
        tkey = ex.get("type") or question_type_key(ex["question"], options)

        # pass1
        out1, probs1_cal, _ = self._score(img, ex["question"], options, short_edge=self.cfg.short_edge_base, extra_prefix="")
        pmax, margin, ent = basic_features(probs1_cal)

        feats = RoutingFeatures(pmax=pmax, margin=margin, entropy=ent, rho=rho, kappa=0.0)

        # candidate pool
        cand = [{
            "name": "pass1",
            "scores": out1.scores,
            "probs_cal": probs1_cal,
            "pred": out1.pred,
            "short_edge": self.cfg.short_edge_base,
        }]

        # routing loop (at most 2 rounds)
        used_cost = 1.0
        used_actions: List[str] = []
        actions_executed: List[str] = []

        routed_to_rag = False
        routed_to_qd = False
        qd_round_preds: List[str] = []
        rag_answers: List[str] = []

        if self.cfg.enable_router and self.router is not None:
            pol = PolicyConfig(
                budget=self.cfg.budget,
                base_cost=1.0,
                max_rounds=self.cfg.router_max_rounds,
                tau=self.cfg.router_tau,
            )

            for _ in range(pol.max_rounds):
                remaining = pol.budget - used_cost
                if remaining <= 1e-9:
                    break

                a = choose_next_action(self.router, feats, pol, remaining_budget=remaining, used_actions=used_actions)
                if a is None:
                    break

                # execute the chosen action
                if a == "HR":
                    used_cost += pol.costs["HR"]
                    out, probs_cal, _ = self._score(img, ex["question"], options, short_edge=self.cfg.short_edge_hr, extra_prefix="")
                    cand.append({"name": "HR", "scores": out.scores, "probs_cal": probs_cal, "pred": out.pred, "short_edge": self.cfg.short_edge_hr})

                    # update state
                    pmax, margin, ent = basic_features(probs_cal)
                    feats = RoutingFeatures(pmax=pmax, margin=margin, entropy=ent, rho=rho, kappa=feats.kappa)

                elif a == "tmRAG":
                    used_cost += pol.costs["tmRAG"]
                    routed_to_rag = True
                    extra = ("[Retrieved examples]\n" + retrieved_block) if retrieved_block else ""
                    out, probs_cal, _ = self._score(img, ex["question"], options, short_edge=self.cfg.short_edge_base, extra_prefix=extra)
                    cand.append({"name": "tmRAG", "scores": out.scores, "probs_cal": probs_cal, "pred": out.pred, "short_edge": self.cfg.short_edge_base})
                    rag_answers = [it.answer for it in retrieved_items]

                    pmax, margin, ent = basic_features(probs_cal)
                    feats = RoutingFeatures(pmax=pmax, margin=margin, entropy=ent, rho=rho, kappa=feats.kappa)

                elif a == "QD":
                    used_cost += pol.costs["QD"]
                    routed_to_qd = True
                    p_hat, kappa, round_preds = self._qd(img, ex["question"], options, desc, retrieved_block, tkey)
                    pred_idx = int(np.argmax(np.asarray(p_hat)))
                    pred = options[pred_idx]
                    qd_round_preds = round_preds
                    cand.append({"name": "QD", "scores": None, "probs_cal": p_hat, "pred": pred, "kappa": kappa})

                    pmax, margin, ent = basic_features(p_hat)
                    feats = RoutingFeatures(pmax=pmax, margin=margin, entropy=ent, rho=rho, kappa=kappa)

                used_actions.append(a)
                actions_executed.append(a)

        # final selection: max calibrated confidence among candidates
        best = max(cand, key=lambda r: float(np.max(np.asarray(r["probs_cal"], dtype=np.float64))))
        final_pred = best["pred"]
        final_probs = best["probs_cal"]
        final_pmax = float(np.max(np.asarray(final_probs, dtype=np.float64)))

        result = {
            "image": ex["image"],
            "question": ex["question"],
            "options": options,
            "gt": gt_text,
            "gt_idx": gt_idx,
            "pass1_pred": out1.pred,
            "final_pred": final_pred,
            "final_pmax": final_pmax,
            "used_budget": used_cost,
            "actions": actions_executed,
            "chosen": best["name"],
            "probs_pass1": probs1_cal,
            "probs_final": final_probs,
            "rho": rho,
            "type_key": tkey,
        }
        if routed_to_rag:
            result["routed_to_rag"] = True
            result["rag_retrieved_answers"] = rag_answers
        if routed_to_qd:
            result["routed_to_qd"] = True
            result["qd_round_preds"] = qd_round_preds
        return result
