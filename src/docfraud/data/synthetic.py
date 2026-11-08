"""Synthetic savings-finance application data with embedded fraud scenarios.

Customers pay a down payment and monthly instalments; at the allocation month the
company finances the home or vehicle. Fraud therefore targets fast allocation
followed by non-payment (bust-out), or fictitious contracts that earn the sales
agent a commission before they are cancelled.

Legitimate noise (households, repeat customers, high earners, shared branch tablets,
genuine defaults) keeps every rule from being a perfect proxy for the label.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from docfraud.validators.tckn import corrupt_tckn, generate_tckn

CITIES = {"Istanbul": 0.30, "Ankara": 0.11, "Izmir": 0.08, "Bursa": 0.05, "Konya": 0.05, "Antalya": 0.05,
          "Kocaeli": 0.04, "Kayseri": 0.04, "Gaziantep": 0.04, "Samsun": 0.04, "Kastamonu": 0.03,
          "Trabzon": 0.04, "Diyarbakir": 0.04, "Eskisehir": 0.04, "Sakarya": 0.05}
OCCUPATIONS = {
    "CIVIL_SERVANT": 48000, "TEACHER": 52000, "WORKER": 32000, "SHOPKEEPER": 60000, "ENGINEER": 90000,
    "HEALTHCARE": 62000, "RETIRED": 26000, "SELF_EMPLOYED": 55000, "PRIVATE_SECTOR": 45000,
}
OCC_W = [0.14, 0.09, 0.20, 0.10, 0.06, 0.07, 0.10, 0.08, 0.16]
CHANNELS = ["BRANCH", "WEB", "MOBILE", "CALL_CENTER"]
CHANNEL_W = [0.58, 0.15, 0.17, 0.10]
EMAIL_DOMAINS = ["gmail.com", "hotmail.com", "outlook.com", "yahoo.com", "icloud.com"]
COMMISSION_RATE = 0.025


@dataclass
class Ctx:
    rng: np.random.Generator
    prng: random.Random
    fake: Faker
    start: datetime
    end: datetime
    branches: pd.DataFrame
    agents: pd.DataFrame
    real_docs: pd.DataFrame
    fake_docs: pd.DataFrame
    seq: int = 0


def iban(prng: random.Random) -> str:
    bank = f"{prng.choice([10, 12, 15, 46, 62, 64, 67, 111, 134, 203, 205, 206, 209, 210]):05d}"
    bban = bank + "0" + "".join(str(prng.randint(0, 9)) for _ in range(16))
    num = int(bban + "292700")
    return f"TR{98 - num % 97:02d}{bban}"


def phone(prng: random.Random) -> str:
    return f"05{prng.choice([30, 31, 32, 33, 35, 36, 38, 39, 42, 43, 44, 45, 46, 50, 51, 52, 53, 54, 55, 59])}" \
           f"{prng.randint(1000000, 9999999)}"


def address(ctx: Ctx, city: str) -> str:
    return (f"{ctx.prng.randint(1, 120)} {ctx.fake.first_name()} Street, Apt {ctx.prng.randint(1, 30)},"
            f" {ctx.fake.last_name()} District, {city}")


def device(prng: random.Random) -> str:
    return "DEV-" + "".join(prng.choice("0123456789ABCDEF") for _ in range(12))


def public_ip(prng: random.Random, prefix: str | None = None) -> str:
    if prefix:
        return f"{prefix}.{prng.randint(2, 254)}"
    return f"{prng.choice([78, 88, 85, 95, 176, 31, 46, 212])}.{prng.randint(0, 255)}." \
           f"{prng.randint(0, 255)}.{prng.randint(2, 254)}"


def make_branches(ctx: Ctx, n: int) -> pd.DataFrame:
    cities = ctx.rng.choice(list(CITIES), n, p=list(CITIES.values()))
    return pd.DataFrame({"branch_id": [f"S{i + 1:03d}" for i in range(n)], "city": cities,
                         "branch_ip_prefix": [f"193.140.{i + 10}" for i in range(n)],
                         "branch_device": [f"BRANCH-TABLET-{i + 1:03d}" for i in range(n)]})


def make_agents(ctx: Ctx, n: int) -> pd.DataFrame:
    br = ctx.rng.choice(ctx.branches.branch_id, n)
    hire = [ctx.start - timedelta(days=int(d)) for d in ctx.rng.integers(30, 2500, n)]
    return pd.DataFrame({"agent_id": [f"T{i + 1:04d}" for i in range(n)], "branch_id": br,
                         "agent_name": [ctx.fake.name() for _ in range(n)], "hire_date": hire})


def month_weights(ctx: Ctx) -> tuple[pd.DatetimeIndex, np.ndarray]:
    months = pd.date_range(ctx.start, ctx.end, freq="MS")
    w = 1.025 ** np.arange(len(months))
    return months, w / w.sum()


def random_ts(ctx: Ctx, lo: datetime | None = None, hi: datetime | None = None, night_prob=0.03) -> datetime:
    if lo is None and hi is None:
        months, w = month_weights(ctx)
        m = months[ctx.rng.choice(len(months), p=w)]
        nxt = min(m + pd.offsets.MonthBegin(1), pd.Timestamp(ctx.end))
        lo, hi = m.to_pydatetime(), nxt.to_pydatetime()
    else:
        lo, hi = lo or ctx.start, min(hi or ctx.end, ctx.end)
    day = lo + timedelta(seconds=float(ctx.rng.uniform(0, max(1, (hi - lo).total_seconds()))))
    hour = int(ctx.rng.integers(0, 6)) if ctx.rng.random() < night_prob else int(ctx.rng.integers(9, 22))
    return day.replace(hour=hour, minute=int(ctx.rng.integers(0, 60)), second=int(ctx.rng.integers(0, 60)))


def person(ctx: Ctx, age_range=(23, 64), occupation=None, income_mult=1.0) -> dict:
    occ = occupation or ctx.rng.choice(list(OCCUPATIONS), p=OCC_W)
    city = ctx.rng.choice(list(CITIES), p=list(CITIES.values()))
    age = int(ctx.rng.integers(*age_range))
    birth = ctx.start - timedelta(days=age * 365 + int(ctx.rng.integers(0, 365)))
    first, last = ctx.fake.first_name(), ctx.fake.last_name()
    income = OCCUPATIONS[occ] * float(np.exp(ctx.rng.normal(0, 0.33))) * income_mult
    tr = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    email = f"{first}.{last}{ctx.prng.randint(1, 99) if ctx.rng.random() < 0.4 else ''}".lower().translate(tr)
    return dict(tckn=generate_tckn(ctx.prng), first_name=first, last_name=last, birth_date=birth.date(),
                occupation=occ, true_income=round(income, -2),
                declared_income=round(income * float(ctx.rng.normal(1.04, 0.06)), -2),
                phone=phone(ctx.prng), email=f"{email}@{ctx.prng.choice(EMAIL_DOMAINS)}",
                iban=iban(ctx.prng), city=city, address=address(ctx, city))


def product(ctx: Ctx, income: float, down_ratio=None, plan=None, fast_alloc=False) -> dict:
    prod = "HOME" if ctx.rng.random() < 0.6 else "VEHICLE"
    term = int(ctx.rng.choice([60, 72, 84, 96, 108, 120] if prod == "HOME" else [24, 36, 48, 60]))
    inst_ratio = float(ctx.rng.uniform(0.15, 0.42))
    installment = round(income * inst_ratio, -2)
    dr = float(np.clip(ctx.rng.beta(2, 9), 0, 0.5)) if down_ratio is None else down_ratio
    financed = installment * term
    amount = round(financed / (1 - dr), -3)
    plan = plan or ("LOTTERY" if ctx.rng.random() < 0.3 else "SCHEDULED")
    if fast_alloc:
        alloc = int(ctx.rng.integers(1, 7))
    elif plan == "SCHEDULED":
        alloc = max(2, int(round(term * ctx.rng.uniform(0.28, 0.45) * (1 - dr))))
    else:
        alloc = int(ctx.rng.integers(1, max(3, int(term * 0.6))))
    return dict(product_type=prod, plan_type=plan, term_months=term, financing_amount=amount,
                down_payment=round(amount * dr, -2), monthly_installment=installment,
                organization_fee=round(amount * 0.08, -2), allocation_month=alloc)


def channel_fields(ctx: Ctx, channel: str, branch: pd.Series, dev=None, ip=None) -> dict:
    if channel in ("BRANCH", "CALL_CENTER"):
        return dict(device_id=branch.branch_device if channel == "BRANCH" else "CALL-CENTER",
                    ip_address=public_ip(ctx.prng, branch.branch_ip_prefix) if channel == "BRANCH" else "10.0.0.1")
    return dict(device_id=dev or device(ctx.prng), ip_address=ip or public_ip(ctx.prng))


def pick_doc(ctx: Ctx, forged: bool) -> dict:
    pool = ctx.fake_docs if forged and len(ctx.fake_docs) else ctx.real_docs
    d = pool.iloc[int(ctx.rng.integers(0, len(pool)))]
    return dict(doc_image_path=d.image_path, doc_type=d.doc_type, doc_is_forged=int(d.label),
                doc_score=float(d.doc_score), doc_top_field=d.top_field,
                mrz_checkdigit_fail=d.get("mrz_checkdigit_fail"), mrz_viz_mismatch=d.get("mrz_viz_mismatch"),
                date_logic_fail=d.get("date_logic_fail"))


def application(ctx: Ctx, p: dict, ts: datetime, scenario: str | None, behaviour: str,
                channel=None, branch_id=None, agent_id=None, forged_doc=False, guarantor=None,
                down_ratio=None, plan=None, fast_alloc=False, dev=None, ip=None) -> dict:
    channel = channel or ctx.rng.choice(CHANNELS, p=CHANNEL_W)
    if agent_id is not None:
        branch_id = ctx.agents.set_index("agent_id").at[agent_id, "branch_id"]
    if branch_id is None:
        same_city = ctx.branches[ctx.branches.city == p["city"]]
        branch_id = (same_city if len(same_city) and ctx.rng.random() < 0.85 else ctx.branches).sample(
            1, random_state=int(ctx.rng.integers(1e9))).branch_id.iloc[0]
    if agent_id is None:
        ag = ctx.agents[ctx.agents.branch_id == branch_id]
        agent_id = (ag if len(ag) else ctx.agents).sample(1, random_state=int(ctx.rng.integers(1e9))).agent_id.iloc[0]
    branch = ctx.branches.set_index("branch_id").loc[branch_id]
    ctx.seq += 1
    return dict(tmp_id=ctx.seq, app_ts=ts, channel=channel, branch_id=branch_id, agent_id=agent_id,
                **{k: v for k, v in p.items() if k != "city"}, applicant_city=p["city"],
                guarantor_tckn=guarantor,
                **product(ctx, p["true_income"], down_ratio, plan, fast_alloc),
                **channel_fields(ctx, channel, branch, dev, ip), **pick_doc(ctx, forged_doc),
                scenario=scenario, behaviour=behaviour)


def legit(ctx: Ctx, n: int) -> list[dict]:
    apps, people = [], []
    for _ in range(n):
        r = ctx.rng.random()
        if people and r < 0.04:
            p = dict(ctx.prng.choice(people))
            base = p["_ts"]
            ts = base + timedelta(days=float(ctx.rng.uniform(1, 25) if ctx.rng.random() < 0.3 else ctx.rng.uniform(30, 300)))
            if ts > ctx.end:
                ts = random_ts(ctx)
        elif people and r < 0.10:
            hh = ctx.prng.choice(people)
            p = person(ctx)
            p.update(address=hh["address"], city=hh["city"])
            if ctx.rng.random() < 0.15:
                p["iban"] = hh["iban"]
            if ctx.rng.random() < 0.10:
                p["phone"] = hh["phone"]
            ts = random_ts(ctx)
        else:
            mult = float(ctx.rng.uniform(3, 5)) if ctx.rng.random() < 0.02 else 1.0
            p = person(ctx, income_mult=mult)
            if mult > 1:
                p["declared_income"] = p["true_income"]
            ts = random_ts(ctx)
        p["_ts"] = ts
        people.append(p)
        q = {k: v for k, v in p.items() if k != "_ts"}
        if ctx.rng.random() < 0.002:
            q["tckn"] = corrupt_tckn(q["tckn"], ctx.prng)
        guarantor = generate_tckn(ctx.prng) if ctx.rng.random() < 0.35 else None
        dr = float(ctx.rng.uniform(0.35, 0.5)) if ctx.rng.random() < 0.05 else None
        u = ctx.rng.random()
        behaviour = "default" if u < 0.025 else "cancel" if u < 0.085 else "normal"
        apps.append(application(ctx, q, ts if ts <= ctx.end else random_ts(ctx), None, behaviour,
                                guarantor=guarantor, down_ratio=dr))
    g = [a for a in apps if a["guarantor_tckn"]]
    for a, b in zip(g[::40], g[1::40]):
        b["guarantor_tckn"] = a["guarantor_tckn"]
    return apps


def forged_id(ctx: Ctx, n: int) -> list[dict]:
    apps = []
    while len(apps) < n:
        p = person(ctx, income_mult=1.3)
        if ctx.rng.random() < 0.2:
            p["tckn"] = corrupt_tckn(p["tckn"], ctx.prng)
        ts = random_ts(ctx, night_prob=0.15)
        repeats = int(ctx.rng.integers(2, 4)) if ctx.rng.random() < 0.3 else 1
        for k in range(repeats):
            t = ts + timedelta(days=float(ctx.rng.uniform(0, 20))) if k else ts
            ch = ctx.rng.choice(CHANNELS, p=[0.5, 0.2, 0.25, 0.05])
            apps.append(application(ctx, dict(p), min(t, ctx.end), "FORGED_ID", "bust_out", channel=ch,
                                    forged_doc=ctx.rng.random() < 0.9, down_ratio=float(ctx.rng.uniform(0.3, 0.5)),
                                    fast_alloc=True))
    return apps[:n]


def synthetic_id(ctx: Ctx, n: int) -> list[dict]:
    apps = []
    for _ in range(n):
        p = person(ctx, age_range=(19, 25))
        p["declared_income"] = round(p["true_income"] * float(ctx.rng.uniform(2.5, 5)), -2)
        if ctx.rng.random() < 0.55:
            p["tckn"] = corrupt_tckn(p["tckn"], ctx.prng)
        p["email"] = f"{''.join(ctx.prng.choice('abcdefghijklmnoprstuvyz') for _ in range(7))}" \
                     f"{ctx.prng.randint(100, 9999)}@{ctx.prng.choice(EMAIL_DOMAINS)}"
        apps.append(application(ctx, p, random_ts(ctx, night_prob=0.25), "SYNTHETIC_ID",
                                ctx.prng.choice(["early_stop", "bust_out"]),
                                channel=ctx.rng.choice(["WEB", "MOBILE"]), forged_doc=ctx.rng.random() < 0.5,
                                fast_alloc=ctx.rng.random() < 0.5))
    return apps


def fraud_ring(ctx: Ctx, n_rings: int) -> list[dict]:
    apps = []
    for ring in range(n_rings):
        size = int(ctx.rng.integers(4, 13))
        phones = [phone(ctx.prng) for _ in range(int(ctx.rng.integers(1, 3)))]
        ibans = [iban(ctx.prng) for _ in range(int(ctx.rng.integers(1, 3)))]
        branch = ctx.branches.sample(1, random_state=int(ctx.rng.integers(1e9))).iloc[0]
        addrs = [address(ctx, branch.city) for _ in range(int(ctx.rng.integers(1, 4)))]
        dev, ip_prefix = device(ctx.prng), f"{ctx.prng.choice([88, 95, 176])}.{ctx.prng.randint(0, 255)}.{ctx.prng.randint(0, 255)}"
        agent = ctx.agents[ctx.agents.branch_id == branch.branch_id]
        agent_id = agent.agent_id.iloc[0] if len(agent) and ctx.rng.random() < 0.6 else None
        start = random_ts(ctx, hi=ctx.end - timedelta(days=50))
        for _ in range(size):
            p = person(ctx, income_mult=float(ctx.rng.uniform(1.5, 3)))
            p["declared_income"] = p["true_income"]
            p["true_income"] = round(p["true_income"] / 2.2, -2)
            p["city"] = branch.city
            if ctx.rng.random() < 0.7:
                p["phone"] = ctx.prng.choice(phones)
            if ctx.rng.random() < 0.6:
                p["iban"] = ctx.prng.choice(ibans)
            if ctx.rng.random() < 0.5:
                p["address"] = ctx.prng.choice(addrs)
            ts = start + timedelta(days=float(ctx.rng.uniform(0, 45)))
            ch = ctx.rng.choice(["BRANCH", "MOBILE", "WEB"], p=[0.6, 0.25, 0.15])
            a = application(ctx, p, min(ts, ctx.end), "FRAUD_RING", ctx.prng.choice(["bust_out", "early_stop"]),
                            channel=ch, branch_id=branch.branch_id, agent_id=agent_id,
                            forged_doc=ctx.rng.random() < 0.35, fast_alloc=ctx.rng.random() < 0.6,
                            dev=dev if ctx.rng.random() < 0.7 else None, ip=public_ip(ctx.prng, ip_prefix))
            a["ring_id"] = f"RING-{ring + 1:03d}"
            apps.append(a)
    return apps


def agent_fraud(ctx: Ctx, n_agents: int) -> list[dict]:
    apps = []
    bad = ctx.agents.sample(n_agents, random_state=int(ctx.rng.integers(1e9)))
    for _, ag in bad.iterrows():
        start = random_ts(ctx, hi=ctx.end - timedelta(days=200))
        for _ in range(int(ctx.rng.integers(30, 51))):
            ts = start + timedelta(days=float(ctx.rng.uniform(0, 160)))
            ts = ts.replace(day=min(28, max(ts.day, int(ctx.rng.integers(22, 29)))))
            p = person(ctx)
            apps.append(application(ctx, p, min(ts, ctx.end), "AGENT_FRAUD", "cancel_after_commission",
                                    channel="BRANCH", agent_id=ag.agent_id, forged_doc=ctx.rng.random() < 0.2))
    return apps


def income_fraud(ctx: Ctx, n: int) -> list[dict]:
    apps = []
    for _ in range(n):
        p = person(ctx)
        p["declared_income"] = round(OCCUPATIONS[p["occupation"]] * float(ctx.rng.uniform(3, 6)), -2)
        a = application(ctx, p, random_ts(ctx), "INCOME_FRAUD", "bust_out",
                        down_ratio=float(ctx.rng.uniform(0.35, 0.5)), fast_alloc=True)
        scale = p["declared_income"] / p["true_income"]
        for k in ("monthly_installment", "financing_amount", "down_payment", "organization_fee"):
            a[k] = round(a[k] * scale, -2)
        apps.append(a)
    return apps


def guarantor_fraud(ctx: Ctx, n_guarantors: int) -> list[dict]:
    apps = []
    for _ in range(n_guarantors):
        g = generate_tckn(ctx.prng)
        if ctx.rng.random() < 0.3:
            g = corrupt_tckn(g, ctx.prng)
        start = random_ts(ctx, hi=ctx.end - timedelta(days=120))
        for _ in range(int(ctx.rng.integers(3, 7))):
            ts = start + timedelta(days=float(ctx.rng.uniform(0, 150)))
            apps.append(application(ctx, person(ctx, income_mult=1.4), min(ts, ctx.end), "GUARANTOR_FRAUD",
                                    "early_stop", guarantor=g, forged_doc=ctx.rng.random() < 0.25))
    return apps


def device_farm(ctx: Ctx, n_devices: int, start: datetime) -> list[dict]:
    apps = []
    for _ in range(n_devices):
        dev = device(ctx.prng)
        prefix = f"{ctx.prng.choice([31, 46, 78])}.{ctx.prng.randint(0, 255)}.{ctx.prng.randint(0, 255)}"
        t0 = random_ts(ctx, lo=start, hi=ctx.end - timedelta(days=10))
        for _ in range(int(ctx.rng.integers(6, 16))):
            ts = t0 + timedelta(days=float(ctx.rng.uniform(0, 10)))
            apps.append(application(ctx, person(ctx, income_mult=1.3), min(ts, ctx.end), "DEVICE_FARM",
                                    "early_stop", channel="MOBILE", forged_doc=ctx.rng.random() < 0.15,
                                    dev=dev, ip=public_ip(ctx.prng, prefix)))
    return apps


def outcomes(ctx: Ctx, apps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pay_rows, contract_rows, label_rows = [], [], []
    end = ctx.end
    for a in apps.itertuples():
        start = a.app_ts
        alloc_date = start + timedelta(days=30.4 * a.allocation_month)
        stop_k, cancel_date = None, None
        if a.behaviour == "bust_out":
            stop_k = a.allocation_month + int(ctx.rng.integers(0, 3))
        elif a.behaviour == "early_stop":
            stop_k = int(ctx.rng.integers(1, 4))
        elif a.behaviour == "default":
            stop_k = int(ctx.rng.integers(3, max(4, a.term_months)))
        elif a.behaviour == "cancel":
            cancel_date = start + timedelta(days=float(ctx.rng.uniform(3, 14) if ctx.rng.random() < 0.6
                                                       else ctx.rng.uniform(60, 400)))
        elif a.behaviour == "cancel_after_commission":
            cancel_date = start + timedelta(days=float(ctx.rng.uniform(45, 100)))
        pay_rows.append((a.application_id, 0, start.date(), start.date(), a.down_payment, "PAID"))
        paid_total = a.down_payment
        for k in range(1, a.term_months + 1):
            due = start + timedelta(days=30.4 * k)
            if due > end or (cancel_date and due > cancel_date):
                break
            if stop_k is not None and k >= stop_k:
                pay_rows.append((a.application_id, k, due.date(), None, a.monthly_installment, "UNPAID"))
                continue
            late = ctx.rng.random() < 0.07
            pd_ = due + timedelta(days=int(ctx.rng.integers(5, 40)) if late else int(ctx.rng.integers(-3, 2)))
            if pd_ > end:
                pay_rows.append((a.application_id, k, due.date(), None, a.monthly_installment, "UNPAID"))
                continue
            pay_rows.append((a.application_id, k, due.date(), pd_.date(), a.monthly_installment,
                             "LATE" if late else "PAID"))
            paid_total += a.monthly_installment
        allocated = (alloc_date <= end and (stop_k is None or stop_k >= a.allocation_month)
                     and (cancel_date is None or alloc_date < cancel_date))
        status = "ACTIVE"
        default_date = None
        if cancel_date and cancel_date <= end:
            status = "CANCELLED"
        elif stop_k is not None and start + timedelta(days=30.4 * (stop_k + 3)) <= end:
            status = "DEFAULTED"
            default_date = (start + timedelta(days=30.4 * (stop_k + 3))).date()
        elif allocated:
            status = "ALLOCATED"
        contract_rows.append(dict(application_id=a.application_id, status=status,
                                  allocation_date=alloc_date.date() if allocated else None,
                                  cancel_date=cancel_date.date() if cancel_date and cancel_date <= end else None,
                                  default_date=default_date, paid_total=round(paid_total, 2),
                                  commission_paid=round(a.financing_amount * COMMISSION_RATE, 2)))
        if isinstance(a.scenario, str):
            if a.behaviour == "cancel_after_commission":
                loss = a.financing_amount * COMMISSION_RATE
                detected = cancel_date + timedelta(days=30)
            elif allocated and status == "DEFAULTED":
                loss = max(0.0, a.financing_amount - paid_total) * 0.6
                detected = start + timedelta(days=30.4 * (stop_k + 3) + 30)
            elif stop_k is not None:
                loss = a.financing_amount * COMMISSION_RATE
                detected = start + timedelta(days=30.4 * (stop_k + 3) + float(ctx.rng.uniform(10, 45)))
            else:
                loss = a.financing_amount * COMMISSION_RATE
                detected = start + timedelta(days=float(ctx.rng.uniform(60, 150)))
            label_rows.append(dict(application_id=a.application_id, is_fraud=1, scenario=a.scenario,
                                   ring_id=getattr(a, "ring_id", None), confirmed_date=detected.date(),
                                   loss_amount=round(loss, 2)))
    payments = pd.DataFrame(pay_rows, columns=["application_id", "installment_no", "due_date", "paid_date",
                                               "amount", "status"])
    return payments, pd.DataFrame(contract_rows), pd.DataFrame(label_rows)


def generate(cfg: dict, doc_pool: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    s = cfg["synthetic"]
    fake = Faker("tr_TR")
    Faker.seed(seed)
    ctx = Ctx(rng=np.random.default_rng(seed), prng=random.Random(seed), fake=fake,
              start=datetime.fromisoformat(s["start_date"]),
              end=datetime.fromisoformat(s["end_date"]).replace(hour=23, minute=59),
              branches=None, agents=None,
              real_docs=doc_pool[doc_pool.label == 0].reset_index(drop=True),
              fake_docs=doc_pool[doc_pool.label == 1].reset_index(drop=True))
    ctx.branches = make_branches(ctx, s["n_branches"])
    ctx.agents = make_agents(ctx, s["n_agents"])

    apps = legit(ctx, s["n_legit_applications"])
    apps += forged_id(ctx, 150)
    apps += synthetic_id(ctx, 120)
    apps += fraud_ring(ctx, 40)
    apps += agent_fraud(ctx, 4)
    apps += income_fraud(ctx, 150)
    apps += guarantor_fraud(ctx, 20)
    apps += device_farm(ctx, 14, datetime.fromisoformat(s["new_fraud_type_start"]))

    df = pd.DataFrame(apps).sort_values("app_ts").reset_index(drop=True)
    df["application_id"] = [f"APP{i + 1:07d}" for i in range(len(df))]
    if "ring_id" not in df:
        df["ring_id"] = None
    payments, contracts, labels = outcomes(ctx, df)

    customers = (df.sort_values("app_ts").groupby("tckn", as_index=False)
                 .first()[["tckn", "first_name", "last_name", "birth_date", "occupation"]])
    documents = df[["application_id", "doc_image_path", "doc_type", "doc_is_forged", "doc_score", "doc_top_field",
                    "mrz_checkdigit_fail", "mrz_viz_mismatch", "date_logic_fail"]].copy()
    app_cols = ["application_id", "app_ts", "channel", "branch_id", "agent_id", "tckn", "first_name", "last_name",
                "birth_date", "occupation", "declared_income", "phone", "email", "iban", "applicant_city", "address",
                "guarantor_tckn", "device_id", "ip_address", "product_type", "plan_type", "term_months",
                "financing_amount", "down_payment", "monthly_installment", "organization_fee", "allocation_month"]
    labels_full = df[["application_id"]].merge(labels, how="left", on="application_id")
    labels_full["is_fraud"] = labels_full.is_fraud.fillna(0).astype(int)
    labels_full["loss_amount"] = labels_full.loss_amount.fillna(0.0)
    return dict(branches=ctx.branches, agents=ctx.agents, customers=customers, applications=df[app_cols],
                documents=documents, payments=payments, contracts=contracts, fraud_labels=labels_full)
