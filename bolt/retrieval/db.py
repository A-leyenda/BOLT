from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RetrievalItem:
    image: str
    question: str
    answer: str
    desc: Optional[Dict[str, str]] = None
    type_key: str = "other"
    options: Optional[List[str]] = None


class RetrievalDB:
    """A simple on-disk tmRAG database.

    Files expected in `db_dir`:
    - embeddings.npy   float32 array [N, D], L2-normalized
    - metas.jsonl      jsonl with aligned rows
    - type_index.json  mapping type_key -> list[int]
    - embed_model.json metadata (optional)

    For simplicity, we load metas and type_index into memory.
    """

    def __init__(self, db_dir: str | Path):
        self.db_dir = Path(db_dir)
        self.emb_path = self.db_dir / "embeddings.npy"
        self.meta_path = self.db_dir / "metas.jsonl"
        self.type_index_path = self.db_dir / "type_index.json"

        self.emb = np.load(self.emb_path, mmap_mode="r")
        self.metas: List[RetrievalItem] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                self.metas.append(
                    RetrievalItem(
                        image=r.get("image", ""),
                        question=r.get("question", ""),
                        answer=r.get("answer", ""),
                        desc=r.get("desc", None),
                        type_key=r.get("type_key", "other"),
                        options=r.get("options", None),
                    )
                )
        self.type_index = json.loads(self.type_index_path.read_text(encoding="utf-8"))
        # convert lists to numpy arrays for faster indexing
        for k, v in list(self.type_index.items()):
            self.type_index[k] = np.asarray(v, dtype=np.int64)

        assert len(self.metas) == self.emb.shape[0], "metas and embeddings length mismatch"

    def retrieve(
        self,
        query_vec: np.ndarray,
        type_key: str,
        topk: int = 4,
        exclude_image: Optional[str] = None,
    ) -> List[Tuple[float, RetrievalItem]]:
        query_vec = np.asarray(query_vec, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-12)

        cand = self.type_index.get(type_key)
        if cand is None or len(cand) == 0:
            cand = np.arange(self.emb.shape[0], dtype=np.int64)

        cand_emb = self.emb[cand]  # [M, D]
        sims = cand_emb @ query_vec  # cosine since normalized
        # topk
        if topk <= 0:
            topk = 1
        k = min(topk * 4, sims.shape[0])  # oversample for exclude_image
        idx_part = np.argpartition(-sims, kth=k-1)[:k]
        idx_sorted = idx_part[np.argsort(-sims[idx_part])]

        out: List[Tuple[float, RetrievalItem]] = []
        for j in idx_sorted:
            global_idx = int(cand[j])
            item = self.metas[global_idx]
            if exclude_image is not None and item.image == exclude_image:
                continue
            out.append((float(sims[j]), item))
            if len(out) >= topk:
                break
        return out
