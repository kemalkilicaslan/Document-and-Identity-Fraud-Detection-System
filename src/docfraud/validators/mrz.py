"""ICAO 9303 TD3 (passport) MRZ parsing and check digits."""
from __future__ import annotations

from datetime import date

WEIGHTS = (7, 3, 1)


def _char_value(c: str) -> int:
    if c.isdigit():
        return int(c)
    if c == "<":
        return 0
    return ord(c.upper()) - 55


def check_digit(field: str) -> str:
    return str(sum(_char_value(c) * WEIGHTS[i % 3] for i, c in enumerate(field)) % 10)


def normalize_line(line: str, length: int = 44) -> str:
    s = "".join(c for c in line.upper().replace(" ", "") if c.isalnum() or c == "<")
    return s[:length].ljust(length, "<")


def parse_td3_line2(line: str) -> dict:
    s = normalize_line(line)
    return dict(document_number=s[0:9], document_number_cd=s[9], nationality=s[10:13],
                birth_date=s[13:19], birth_date_cd=s[19], sex=s[20], expiry_date=s[21:27],
                expiry_date_cd=s[27], optional=s[28:42], optional_cd=s[42], composite_cd=s[43])


def parse_td3_line1(line: str) -> dict:
    s = normalize_line(line)
    names = s[5:].split("<<", 1)
    return dict(doc_code=s[0:2], issuing_state=s[2:5],
                surname=names[0].replace("<", " ").strip(),
                given_names=names[1].replace("<", " ").strip() if len(names) > 1 else "")


def check_td3_line2(line: str) -> dict[str, bool]:
    """Which check digits of the second MRZ line are consistent."""
    s = normalize_line(line)
    p = parse_td3_line2(s)
    composite = s[0:10] + s[13:20] + s[21:43]
    res = {
        "document_number": check_digit(p["document_number"]) == p["document_number_cd"],
        "birth_date": check_digit(p["birth_date"]) == p["birth_date_cd"],
        "expiry_date": check_digit(p["expiry_date"]) == p["expiry_date_cd"],
        "composite": check_digit(composite) == p["composite_cd"],
    }
    if p["optional"].strip("<"):
        res["optional"] = check_digit(p["optional"]) == p["optional_cd"]
    return res


def mrz_date(yymmdd: str, future: bool = False) -> date | None:
    """YYMMDD -> date. Birth dates are in the past; expiry dates may be in the future."""
    if len(yymmdd) != 6 or not yymmdd.isdigit():
        return None
    yy, mm, dd = int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:])
    year = 2000 + yy if (future or yy <= 30) else 1900 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None
