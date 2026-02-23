from __future__ import annotations

import ast
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


_WS_RE = re.compile(r"\s+")
_PUNCT_TAIL_RE = re.compile(r"[\s\.,!\?:;]+$")


def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = _PUNCT_TAIL_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    return s.lower()


def extract_qa_from_conversations(example: Dict[str, Any]) -> Tuple[str, str]:
    """Extract (question, answer) from a LLaVA-style `conversations` field.

    We treat the last human/user turn as question and the last assistant/gpt turn as answer.
    """
    conv = example.get("conversations") or example.get("conversation") or []
    q, a = "", ""
    if isinstance(conv, list):
        for m in conv:
            if not isinstance(m, dict):
                continue
            frm = (m.get("from") or m.get("role") or "").lower()
            val = m.get("value") or m.get("content") or ""
            if frm in {"human", "user"}:
                q = val
            elif frm in {"gpt", "assistant", "bot"}:
                a = val
    q = (q or "").replace("<image>", "").strip()
    a = (a or "").strip()
    return q, a


def parse_options_from_question(question: str) -> Optional[List[str]]:
    """Try to parse options embedded in question text.

    Supports patterns like:
    - "C: ['A', 'B', 'C']"
    - "Choices: [ ... ]"

    Returns None if cannot parse.
    """
    q = question or ""

    # common "C:" style
    m = re.search(r"(?:^|\n)\s*(?:C|Choices)\s*:\s*(\[[^\]]+\])\s*(?:$|\n)", q)
    if m:
        raw = m.group(1).strip()
        try:
            opts = ast.literal_eval(raw)
            if isinstance(opts, list) and all(isinstance(x, str) for x in opts):
                return [o.strip() for o in opts]
        except Exception:
            pass

    # bullet list like:
    # A) ...
    # B) ...
    # We only return the option *labels* (A/B/...) if present consistently.
    m2 = re.findall(r"^(\s*[A-E])\s*[\)\.]\s+.+$", q, flags=re.MULTILINE)
    if m2 and len(set(m2)) >= 2:
        return sorted(set([x.strip() for x in m2]))

    return None


def choose_options(example: Dict[str, Any], question: str) -> Optional[List[str]]:
    """Get options either from explicit field or parse from question."""
    opts = example.get("options")
    if isinstance(opts, list) and opts and all(isinstance(x, str) for x in opts):
        return [o.strip() for o in opts]
    return parse_options_from_question(question)


def match_option_index(options: List[str], answer: str) -> Optional[int]:
    """Return the index of answer in options (string-normalized)."""
    a = norm_text(answer)
    if not a:
        return None
    for i, o in enumerate(options):
        if norm_text(o) == a:
            return i
    return None


def question_type_key(question: str, options: Optional[List[str]] = None) -> str:
    """Heuristic type key used for type-matched retrieval (tmRAG).

    This is *not* required if your dataset provides explicit types.
    """
    q = (question or "").lower()
    opts = options or []

    # yes/no
    if len(opts) == 2 and set(map(norm_text, opts)) == {"yes", "no"}:
        return "yn"

    if "colored arrow" in q:
        return "arrows"
    if "colored point" in q:
        return "colors"
    if "goal state" in q or "configuration" in q:
        return "letters"

    # option-set based fallback
    opt_norm = set(map(norm_text, opts))
    if opt_norm.issuperset({"red", "blue", "green"}) and "none of the above" in opt_norm:
        return "colors"

    return "other"
