"""Turkish national ID number (TCKN) check digits.

11 digits, first digit not 0,
  d10 = ((d1+d3+d5+d7+d9) * 7 - (d2+d4+d6+d8)) mod 10
  d11 = (d1+...+d10) mod 10
"""
from __future__ import annotations

import random


def tckn_check_digits(first9: str) -> str:
    d = [int(c) for c in first9]
    d10 = ((sum(d[0:9:2]) * 7) - sum(d[1:8:2])) % 10
    d11 = (sum(d) + d10) % 10
    return f"{d10}{d11}"


def is_valid_tckn(value) -> bool:
    s = str(value).strip() if value is not None else ""
    if len(s) != 11 or not s.isdigit() or s[0] == "0":
        return False
    return s[9:] == tckn_check_digits(s[:9])


def generate_tckn(rng: random.Random | None = None) -> str:
    rng = rng or random
    first9 = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(8))
    return first9 + tckn_check_digits(first9)


def corrupt_tckn(tckn: str, rng: random.Random | None = None) -> str:
    """Change one of the last two digits so the checksum fails."""
    rng = rng or random
    pos = rng.choice([9, 10])
    new = str((int(tckn[pos]) + rng.randint(1, 9)) % 10)
    return tckn[:pos] + new + tckn[pos + 1:]
