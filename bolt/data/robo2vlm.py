from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

from bolt.utils.io import iter_jsonl
from bolt.utils.text import choose_options, extract_qa_from_conversations


def _walk_json(obj: Any) -> Generator[Dict[str, Any], None, None]:
    if isinstance(obj, dict):
        # If dict looks like an example
        if any(k in obj for k in ["image", "image_id", "file_name", "filename", "conversations", "question", "q"]):
            yield obj
        # else traverse
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for it in obj:
            yield from _walk_json(it)


def iter_examples_any(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    """Iterate over examples from a JSON or JSONL file.

    Supports:
    - JSON list/dict (nested ok)
    - JSONL (one dict per line)
    """
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        yield from iter_jsonl(path)
        return
    # json
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    yield from _walk_json(obj)


def get_image_name(example: Dict[str, Any]) -> Optional[str]:
    for k in ["image", "image_name", "image_id", "file_name", "filename", "file", "name"]:
        v = example.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def normalize_example(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize a raw example into a common format used by our pipeline."""
    img = get_image_name(example)
    if not img:
        return None

    # question & answer
    q = example.get("q") or example.get("question")
    a = example.get("a") or example.get("answer")
    if not isinstance(q, str) or not q.strip():
        # try conversations
        q2, a2 = extract_qa_from_conversations(example)
        q = q2
        if not a and a2:
            a = a2
    if not isinstance(q, str) or not q.strip():
        return None

    if not isinstance(a, str):
        a = a or ""

    # options
    opts = choose_options(example, q)

    out = {
        "image": img,
        "question": q.strip(),
        "answer": a.strip(),
        "options": opts or None,
    }
    # preserve original id/type if provided
    if "id" in example:
        out["id"] = example["id"]
    if "type" in example:
        out["type"] = example["type"]
    return out
