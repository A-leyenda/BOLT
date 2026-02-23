from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

from PIL import Image


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def iter_jsonl(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    """Streaming JSONL reader."""
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_open_image(path: str | Path) -> Image.Image:
    """Open image as RGB."""
    return Image.open(path).convert("RGB")


def expanduser(path: str | Path) -> str:
    return os.path.expandvars(os.path.expanduser(str(path)))
