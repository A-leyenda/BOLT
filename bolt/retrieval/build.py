from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from bolt.data.robo2vlm import iter_examples_any, normalize_example
from bolt.decoding.generation import hf_generate
from bolt.decoding.prompts import DEFAULT_DESC_PROMPT
from bolt.models.peft_utils import load_student_with_adapter
from bolt.utils.io import ensure_dir, safe_open_image
from bolt.utils.text import question_type_key


def _normalize_desc(d: Any) -> Optional[Dict[str, str]]:
    if d is None:
        return None
    if isinstance(d, dict):
        en = str(d.get("en", "") or "")
        zh = str(d.get("zh", "") or "")
        if en or zh:
            return {"en": en, "zh": zh}
        return None
    if isinstance(d, str) and d.strip():
        # store in en field
        return {"en": d.strip(), "zh": ""}
    return None


def _make_text_for_embed(question: str, desc: Optional[Dict[str, str]]) -> str:
    d_en = (desc or {}).get("en", "")
    d_zh = (desc or {}).get("zh", "")
    # Keep it short to stabilize embeddings.
    d_en = d_en[:256]
    d_zh = d_zh[:256]
    return f"Q: {question.strip()}\nEN: {d_en}\nZH: {d_zh}".strip()


@dataclass
class BuildIndexConfig:
    data_jsonl: str
    image_root: str
    out_dir: str
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    desc_mode: str = "none"  # none | dataset | generate
    student_ckpt: Optional[str] = None  # required if desc_mode=generate
    batch_size: int = 128
    max_samples: int = -1


def build_retrieval_index(cfg: BuildIndexConfig) -> None:
    out_dir = ensure_dir(cfg.out_dir)

    # sentence embedding model
    from sentence_transformers import SentenceTransformer

    # Default to CPU for SBERT (VLM usually uses GPU; avoid contention).
    try:
        sbert = SentenceTransformer(cfg.embed_model, device="cpu")
    except Exception:
        sbert = SentenceTransformer(cfg.embed_model)

    dim = int(sbert.get_sentence_embedding_dimension())

    # optional: load student for description generation
    gen_model = gen_processor = None
    if cfg.desc_mode == "generate":
        if not cfg.student_ckpt:
            raise ValueError("desc_mode=generate requires --student_ckpt")
        loaded = load_student_with_adapter(cfg.student_ckpt, device_map="auto", torch_dtype="auto", load_in_4bit=False)
        gen_model, gen_processor = loaded.model, loaded.processor
        gen_model.eval()

    # first pass: count eligible examples
    n_total = 0
    for raw in iter_examples_any(cfg.data_jsonl):
        ex = normalize_example(raw)
        if ex is None:
            continue
        if not ex.get("options"):
            continue
        n_total += 1
        if cfg.max_samples > 0 and n_total >= cfg.max_samples:
            break
    if n_total == 0:
        raise RuntimeError("No eligible examples found (need options).")

    emb_path = out_dir / "embeddings.npy"
    meta_path = out_dir / "metas.jsonl"
    type_index_path = out_dir / "type_index.json"
    info_path = out_dir / "embed_model.json"

    emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float32, shape=(n_total, dim))

    type_index: Dict[str, List[int]] = {}

    # second pass: encode in chunks
    buf_texts: List[str] = []
    buf_indices: List[int] = []
    i = 0

    meta_f = meta_path.open("w", encoding="utf-8")

    def flush():
        nonlocal buf_texts, buf_indices
        if not buf_texts:
            return
        vecs = sbert.encode(buf_texts, batch_size=cfg.batch_size, show_progress_bar=False, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        for j, idx in enumerate(buf_indices):
            emb[idx] = vecs[j]
        buf_texts = []
        buf_indices = []

    for raw in tqdm(iter_examples_any(cfg.data_jsonl), desc="Build tmRAG DB"):
        ex = normalize_example(raw)
        if ex is None:
            continue
        opts = ex.get("options") or []
        if not opts:
            continue

        # description
        desc = None
        if cfg.desc_mode == "dataset":
            desc = _normalize_desc(raw.get("desc") or raw.get("description") or raw.get("caption"))
        elif cfg.desc_mode == "generate":
            img_path = Path(cfg.image_root) / ex["image"]
            if img_path.exists():
                img = safe_open_image(img_path)
                txt = hf_generate(gen_model, gen_processor, img, DEFAULT_DESC_PROMPT, max_new_tokens=256, min_new_tokens=16, do_sample=False)
                # best-effort: store raw text (can be post-processed into JSON by user)
                desc = {"en": txt, "zh": ""}
        else:
            desc = None

        tkey = ex.get("type") or question_type_key(ex["question"], opts)
        type_index.setdefault(tkey, []).append(i)

        meta_f.write(json.dumps({
            "image": ex["image"],
            "question": ex["question"],
            "answer": ex.get("answer", ""),
            "options": opts,
            "type_key": tkey,
            "desc": desc,
        }, ensure_ascii=False) + "\n")

        buf_texts.append(_make_text_for_embed(ex["question"], desc))
        buf_indices.append(i)
        i += 1

        if len(buf_texts) >= 4096:
            flush()

        if cfg.max_samples > 0 and i >= cfg.max_samples:
            break

    flush()
    meta_f.close()
    emb.flush()

    info_path.write_text(json.dumps({"embed_model": cfg.embed_model, "dim": dim, "desc_mode": cfg.desc_mode}, indent=2), encoding="utf-8")
    type_index_path.write_text(json.dumps(type_index), encoding="utf-8")

    print(f"[tmRAG] DB saved to: {out_dir}")
    print(f"[tmRAG] N={i}  dim={dim}  types={len(type_index)}")
