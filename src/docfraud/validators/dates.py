"""Date parsing for the visual zone of SIDTD documents and date-logic checks."""
from __future__ import annotations

import re
from datetime import date

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
MONTHS["OKT"] = 10  # Greek passports print "Okt"


def _year(y: str) -> int:
    v = int(y)
    if len(y) == 2:
        v += 2000 if v <= 40 else 1900
    return v


def parse_date(text: str) -> date | None:
    """Parse the date formats used in SIDTD: 20-11-1991, 14.08.1994., 31 01 1971,
    01. 04. 1981, 02 Jan 87, 140915 (ddmmyy)."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"(\d{1,2})\s*[.\-/ ]\s*(\d{1,2})\s*[.\-/ ]\s*(\d{4}|\d{2})", t)
    if m:
        d, mo, y = m.groups()
    else:
        m = re.search(r"(\d{1,2})\s*([A-Z]{3})\w*\s*(\d{4}|\d{2})", t)
        if m and m.group(2) in MONTHS:
            d, mo, y = m.group(1), str(MONTHS[m.group(2)]), m.group(3)
        else:
            m = re.fullmatch(r"(\d{2})(\d{2})(\d{2})", re.sub(r"\D", "", t))
            if not m:
                return None
            d, mo, y = m.groups()
    try:
        return date(_year(y), int(mo), int(d))
    except ValueError:
        return None


def date_logic_flags(birth: date | None, issue: date | None, expiry: date | None,
                     min_age: int = 0, max_validity_years: int = 15) -> dict[str, bool]:
    """True means the check FAILED (suspicious)."""
    flags = {}
    if birth and issue:
        flags["issue_before_birth"] = issue <= birth
        flags["issued_under_min_age"] = (issue.year - birth.year) < min_age
    if issue and expiry:
        flags["expiry_before_issue"] = expiry <= issue
        flags["validity_too_long"] = (expiry.year - issue.year) > max_validity_years
    if birth:
        flags["birth_in_future"] = birth > date.today()
    return flags
