"""Shared script setup: puts src/ on sys.path and times each step."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_t0 = time.time()


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def done() -> None:
    print(f"--- done in {time.time() - _t0:.1f}s", flush=True)
