"""Config loading and project paths."""
from __future__ import annotations

import os
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict:
    cfg_file = os.environ.get("DOCFRAUD_CONFIG", ROOT / "configs" / "config.yaml")
    with open(cfg_file, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in config.yaml (created if it is a dir)."""
    p = ROOT / load_config()["paths"][key]
    if not p.suffix:
        p.mkdir(parents=True, exist_ok=True)
    return p


def seed_everything(seed: int | None = None) -> int:
    seed = load_config()["seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


def rel(p: Path) -> str:
    """Path relative to the project root when possible (portable tables), else absolute."""
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
