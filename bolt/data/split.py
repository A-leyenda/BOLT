from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any, Dict, List, Tuple

from bolt.data.robo2vlm import normalize_example


@dataclass
class SplitResult:
    train_kd: List[Dict[str, Any]]
    val: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


def split_by_image_id(
    train_raw: List[Dict[str, Any]],
    test_raw: List[Dict[str, Any]],
    val_size: int,
    seed: int = 42,
) -> SplitResult:
    """Split train into (train_kd, val) by **image id** to prevent leakage.

    The test split is taken from test_raw, and we also enforce no image overlap with train/val.

    Args:
        train_raw: raw train examples (dicts)
        test_raw: raw test examples (dicts)
        val_size: target number of *items* in val (we add images until we reach >= val_size)
        seed: RNG seed
    """
    rng = random.Random(seed)

    train = [normalize_example(e) for e in train_raw]
    train = [e for e in train if e is not None]

    test = [normalize_example(e) for e in test_raw]
    test = [e for e in test if e is not None]

    # group train by image
    img2items = defaultdict(list)
    for e in train:
        img2items[e["image"]].append(e)

    img_ids = list(img2items.keys())
    rng.shuffle(img_ids)

    val_imgs = []
    val_items: List[Dict[str, Any]] = []
    for img in img_ids:
        if len(val_items) >= val_size:
            break
        val_imgs.append(img)
        val_items.extend(img2items[img])

    val_img_set = set(val_imgs)
    train_kd_items = [e for e in train if e["image"] not in val_img_set]

    # enforce test disjointness (drop any overlapping images)
    train_img_set = set([e["image"] for e in train_kd_items]) | val_img_set
    test_items = [e for e in test if e["image"] not in train_img_set]

    return SplitResult(train_kd=train_kd_items, val=val_items, test=test_items)
