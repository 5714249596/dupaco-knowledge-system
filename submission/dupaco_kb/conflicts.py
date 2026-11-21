"""Contradiction finder.

A *topic* is a product + its qualifiers (term, tier, jumbo, purchase/refinance ...). Every rate row
seen on any page in any snapshot is mapped to a topic. A topic is a conflict when the website has
shown different values for it — on two pages at the same time (a stale calculator / old landing page
next to the official rate sheet) or on the official page across snapshots (the rate moved but the old
number is still quoted somewhere or was quoted before).

Resolution = the value on the product's official page (or the central /rates sheet) in the newest
snapshot that still lists the topic. Qualifiers are part of the topic, so tiers never conflict with
each other.
"""
from __future__ import annotations

import re
from collections import defaultdict

from . import text as T
from .authority import family_home

_NUM = re.compile(r"(\d+(?:\.\d+)?)\s?%")


def _pcts(values) -> list:
    out = []
    for h, v in values:
        for m in _NUM.finditer(v):
            out.append((h, m.group(1)))
    return out


def _fmt(x: str) -> str:
    return f"{x}%"


def _primary(row, pcts, fam=""):
    """The value a member would quote: APY for deposits, APR for loans (note rate for mortgages)."""
    order = ("rate", "APR") if fam == "mortgage" else ("APY", "APR")
    for want in order:
        for h, v in pcts:
            if h == want:
                return v
    return pcts[0][1] if pcts else None


def topic_of(row):
    """Map a row to (family, topic label, sort tier) or None."""
    lab = T.norm_dash(row.label)
    ll = lab.lower()
    ctx = row.context.lower()
    path = row.path.lower()
    terms = sorted(row.terms)
    if re.match(r"(roth|traditional)\b", ll) or "retirement" in path:
        return None
    if row.myr and terms:
        if "recreational" in path or re.search(r"model year (20\d\d)-(\d\d)\b", ll) and \
                int(re.search(r"model year (20\d\d)-(\d\d)\b", ll).group(1)[2:]) > int(re.search(r"model year (20\d\d)-(\d\d)\b", ll).group(2)):
            return None      # RV / motorcycle tables ("Model Year 2026-23") are a different product
        return ("auto", f"{terms[0]}-month auto loan APR", -row.myr[1])
    m = re.search(r"(purchase|refinance)?:?\s*(\d{2})-year fixed", ll)
    if m and ("conventional" in ll or "fixed" in ll):
        refi = 0.5 if m.group(1) == "refinance" else 0.0      # purchase and refinance kept as separate tiers
        cap = T.balance_range(lab)
        return ("mortgage", f"{m.group(2)}-year fixed mortgage rate", (cap[1] if cap else 1e9) + refi)
    hdr = " ".join(row.header).lower()
    if ("certificate" in ll or "certificate" in ctx or "certificate" in hdr or _CERT in row.toks) \
            and len(terms) == 1 and (not row.is_line or "certificate" in ll):
        jumbo = "jumbo " if "jumbo" in ll else ""
        special = " special" if "special" in ll else ""
        return ("certificate", f"{terms[0]}-month {jumbo}certificate{special} APY", 0)
    if "visa" in ll and not row.is_line:
        card = re.sub(r"\s*(rate|standard|:.*)$", "", lab.split("|")[0].strip(), flags=re.I).strip()
        if "introductory" in ll or "intro" in ll or "balance transfer" in ll:
            return None
        return ("credit_card", f"{card} APR", 0)
    if "heloc" in ll or "home equity line" in ll:
        return ("heloc", "HELOC APR", 0)
    if row.bal and not row.is_line:
        prod = lab.split("|")[0].strip()
        prod = re.sub(r"\s*\((business)\)", "", prod, flags=re.I)
        if not re.search(r"saving|checking|money market|club|account", prod, re.I):
            if re.search(r"checking", path + ctx) and "$" in prod:
                prod = "1st Rate Checking"
            else:
                return None
        lo, hi = row.bal
        tier = f"${lo:,.0f}+" if hi == float("inf") else f"${lo:,.0f}-${hi:,.0f}"
        return ("deposit", f"{prod} {tier} APY", lo)
    if "loans/personal-loans" in path and terms and not row.is_line:
        return ("personal_loan", f"{terms[0]}-month personal loan APR", 0)
    return None


_CERT = T.stem("certificate")
_EXPIRED = re.compile(r"\b(ended|expired|no longer (?:available|offered))\b", re.I)

_FAMS_FOR = {"auto": "auto", "mortgage": "mortgage", "certificate": "certificate", "credit_card": "credit_card",
             "heloc": "heloc", "deposit": "savings", "personal_loan": "personal_loan"}


def _official(fam: str, path: str) -> bool:
    f = _FAMS_FOR.get(fam, fam)
    if family_home(f, path):
        return True
    if fam == "deposit":
        return family_home("checking", path) or family_home("money_market", path)
    return False


def find_conflicts(store) -> list[dict]:
    if not store.views:
        return []
    obs = defaultdict(list)       # topic -> [(view_idx, sid, url, path, kind, official, primary, all values, tier)]
    for vi, v in enumerate(store.views):
        for r in v.rows:
            t = topic_of(r)
            if not t:
                continue
            fam, topic, tier = t
            pcts = [(h, x) for h, x in _pcts(r.values) if float(x) > 0 or fam == "deposit"]
            if not pcts:
                continue
            prim = _primary(r, pcts, fam)
            expired = bool(_EXPIRED.search(r.label + " " + r.context))
            official = _official(fam, r.path) and not expired
            obs[(fam, topic)].append((vi, v.sid, r.url, r.path, r.kind, official, prim,
                                      tuple(sorted({x for _, x in pcts}, key=float)), tier, expired))
    # auto tiers: compare by *rank* of the model-year band on the same page ("newest band" = 0),
    # because the band labels roll forward each year (2023-26 -> 2024-27)
    for (fam, topic), items in obs.items():
        if fam != "auto":
            continue
        bands = defaultdict(set)
        for o in items:
            bands[(o[0], o[2])].add(o[8])
        for i, o in enumerate(items):
            rank = sorted(bands[(o[0], o[2])]).index(o[8])
            items[i] = o[:8] + (rank,) + o[9:]
    last = len(store.views) - 1
    out = []
    for (fam, topic), items in obs.items():
        if len({o[7] for o in items}) < 2:
            continue
        official = [o for o in items if o[5]]
        if not official:
            # site without recognisable product pages: trust first-party pages that are not expired promos
            official = [o for o in items if o[4] in ("rates", "core") and not o[9]]
        if not official:
            continue
        newest_vi = max(o[0] for o in official)
        cur = [o for o in official if o[0] == newest_vi]
        # among model-year / loan-amount tiers keep the headline tier (newest model year, smallest loan)
        best_tier = min(o[8] for o in cur) if fam == "auto" else None
        if fam == "mortgage":
            best_tier = min(o[8] for o in cur)
        if best_tier is not None:
            cur = [o for o in cur if o[8] == best_tier] or cur
        cur.sort(key=lambda o: (0 if o[3] == "rates" else 1, o[3]))
        winner = cur[0][6]
        wtuple = cur[0][7]
        stale_now = [o for o in items if o[0] == last and o[7] != wtuple and o[8] == cur[0][8]]
        changed = len({o[7] for o in official if o[8] == cur[0][8]}) > 1
        if not stale_now and not changed:
            continue
        # reason
        if stale_now:
            where = sorted({o[3] or "home" for o in stale_now})[:3]
            reason = (f"{', '.join(where)} still show {', '.join(_fmt(x) for x in sorted({o[6] for o in stale_now}, key=float))} "
                      f"in the newest snapshot; the official rate page shows {_fmt(winner)} and is authoritative.")
        elif newest_vi < last:
            reason = (f"The official page last listed {_fmt(winner)} (snapshot {store.views[newest_vi].sid}); "
                      f"older values came from earlier or less authoritative pages.")
        else:
            reason = (f"The rate changed across snapshots ({' -> '.join(_fmt(x) for x in _timeline(official))}); "
                      f"the newest official value {_fmt(winner)} wins.")
        if fam == "mortgage":      # every loan-amount tier of the same purchase/refinance program
            rel = [o for o in items if (o[8] % 1) == (cur[0][8] % 1)]
        else:                      # same tier as the winner only
            rel = [o for o in items if o[8] == cur[0][8]]
        values = [_fmt(x) for x in sorted({o[6] for o in rel}, key=float)]
        # also list secondary values (e.g. dividend rate vs APY, rate vs APR) seen for this topic
        extra = sorted({x for o in rel for x in o[7]} - {o[6] for o in rel}, key=float)
        items = rel
        srcs, seen = [], set()
        for o in sorted(items, key=lambda o: (-o[0], not o[5], o[3])):
            k = (o[2], o[6])
            if k in seen:
                continue
            seen.add(k)
            srcs.append((o[2], o[1], f"{topic}: {_fmt(o[6])}"))
            if len(srcs) >= 6:
                break
        prio = (3 if stale_now else 0) + (2 if changed else 0) + (1 if fam in ("auto", "mortgage", "certificate") else 0)
        out.append({
            "topic": topic, "values": values + [_fmt(x) for x in extra[:6]],
            "sources": srcs, "resolution": _fmt(winner), "reason": reason, "_prio": prio,
        })
    # tiers of one product that moved together (same values) are one contradiction, not several
    merged, index = [], {}
    for c in sorted(out, key=lambda c: c["topic"]):
        base = re.sub(r"\s*\$[\d,]+(?:\+|-\$[\d,]+)\s*", " ", c["topic"]).strip()
        k = (base, tuple(c["values"]), c["resolution"])
        if k in index:
            continue
        index[k] = c
        if base != c["topic"] and sum(1 for x in out if re.sub(r"\s*\$[\d,]+(?:\+|-\$[\d,]+)\s*", " ", x["topic"]).strip() == base
                                      and tuple(x["values"]) == k[1]) > 1:
            c["topic"] = base + " (several balance tiers)"
        merged.append(c)
    out = merged
    out.extend(_prose_conflicts(store))
    out.sort(key=lambda c: (-c["_prio"], c["topic"]))
    for c in out:
        c.pop("_prio", None)
    return out[:60]


def _timeline(items):
    seq = []
    for o in sorted(items, key=lambda o: o[0]):
        if not seq or seq[-1] != o[6]:
            seq.append(o[6])
    return seq


_ORD = "first second third fourth fifth sixth seventh eighth ninth tenth".split()
PROSE_PATTERNS = [
    ("number of Dupaco branch locations",
     re.compile(r"(?<![\d,.])\b(\d{1,3})\s+(?:full[- ]service\s+)?branch(?:es| office locations| offices| locations)\b", re.I)),
    ("times Forbes has recognized Dupaco",
     re.compile(r"\b(" + "|".join(_ORD) + r") time Forbes\b", re.I)),
]


def _prose_conflicts(store) -> list[dict]:
    v = store.views[-1]
    out = []
    for topic, rx in PROSE_PATTERNS:
        seen = defaultdict(list)    # value -> [(pub ordinal, url)]
        for pg in v.pages.values():
            for m in rx.finditer(pg.md):
                val = m.group(1).lower()
                if val.isdigit() and int(val) > 200:
                    continue
                seen[val].append((pg.pub.toordinal() if pg.pub else 0, pg.url))
        if len(seen) < 2:
            continue
        newest = max(seen, key=lambda k: max(o for o, _ in seen[k]))
        srcs = []
        for val, lst in sorted(seen.items()):
            o, url = max(lst)
            srcs.append((url, v.sid, f"{topic}: {val}"))
        out.append({
            "topic": topic, "values": sorted(seen, key=lambda x: (len(x), x)), "sources": srcs[:6],
            "resolution": newest,
            "reason": "Older press releases repeat the figure from their publication date; the most recently "
                      "published page is taken as current.",
            "_prio": 4,
        })
    return out
