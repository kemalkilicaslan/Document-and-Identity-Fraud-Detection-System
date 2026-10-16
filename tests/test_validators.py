import random
from datetime import date

from docfraud.validators.dates import date_logic_flags, parse_date
from docfraud.validators.mrz import check_digit, check_td3_line2, mrz_date, parse_td3_line1
from docfraud.validators.tckn import corrupt_tckn, generate_tckn, is_valid_tckn

GOOD_LINES = [
    "AK69955741GRC8701026M2303174<<<<<<<<<<<<<<02",
    "LV63090383LVA7409288M2611044280974<14045<<02",
    "C193895647AZE9408148M28081525188L2V<<<<<<<42",
]


def test_known_tckn():
    assert is_valid_tckn("10000000146")
    assert not is_valid_tckn("10000000147")
    assert not is_valid_tckn("01234567890")
    assert not is_valid_tckn("1234567890")
    assert not is_valid_tckn("1000000014a")
    assert not is_valid_tckn(None)


def test_generated_tckn_round_trip():
    rng = random.Random(0)
    for _ in range(500):
        t = generate_tckn(rng)
        assert is_valid_tckn(t)
        assert not is_valid_tckn(corrupt_tckn(t, rng))


def test_mrz_check_digit_icao_example():
    assert check_digit("L898902C3") == "6"
    assert check_digit("740812") == "2"


def test_mrz_lines_valid_and_tampered():
    for line in GOOD_LINES:
        assert all(check_td3_line2(line).values()), line
    tampered = GOOD_LINES[0][:13] + "8801026" + GOOD_LINES[0][20:]
    res = check_td3_line2(tampered)
    assert not res["birth_date"] and not res["composite"]


def test_mrz_line1_and_dates():
    p = parse_td3_line1("P<GRCPAPAGO<<GABRIEL<<<<<<<<<<<<<<<<<<<<<<<<")
    assert p["surname"] == "PAPAGO" and p["given_names"] == "GABRIEL"
    assert mrz_date("870102") == date(1987, 1, 2)
    assert mrz_date("230317", future=True) == date(2023, 3, 17)


def test_parse_sidtd_date_formats():
    assert parse_date("20-11-1991") == date(1991, 11, 20)
    assert parse_date("14.08.1994.") == date(1994, 8, 14)
    assert parse_date("31 01 1971") == date(1971, 1, 31)
    assert parse_date("01. 04. 1981") == date(1981, 4, 1)
    assert parse_date("02 Jan 87") == date(1987, 1, 2)
    assert parse_date("25 Okt 19") == date(2019, 10, 25)
    assert parse_date("140915") == date(2015, 9, 14)
    assert parse_date("garbage") is None


def test_date_logic():
    ok = date_logic_flags(date(1990, 1, 1), date(2015, 1, 1), date(2025, 1, 1))
    assert not any(ok.values())
    bad = date_logic_flags(date(1990, 1, 1), date(2015, 1, 1), date(2014, 1, 1))
    assert bad["expiry_before_issue"]
