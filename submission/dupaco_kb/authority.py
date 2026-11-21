"""Which pages to trust. Rules are written down in NOTES.md.

Every page gets a *kind* and a base weight. Official product / rate pages beat landing pages,
which beat PDFs, blogs, newsroom posts and calculators (calculators are notorious for keeping
old rates). For rate questions we also know which page is the *home* of each product family:
the auto-loan rate is whatever the auto-loan page says, not what the checking page's stale
embedded rate block says.
"""
from __future__ import annotations

import re

KIND_WEIGHT = {
    "rates": 1.0,     # official product / rate pages
    "core": 0.9,      # other first-party pages (contact, about, landing pages)
    "feesheet": 0.95,  # the official fee schedule PDF
    "news": 0.6,
    "blog": 0.5,
    "doc": 0.4,       # other PDFs / spreadsheets
    "calc": 0.25,     # calculators: illustrative, often stale
    "other": 0.6,
}

RATE_PAGE = re.compile(
    r"^(personal/(savings-account|checking-accounts|credit-cards|retirement|loans/[a-z-]+|savings/[a-z-]+)|"
    r"business-rates|rates|business/(checking-accounts|business-checking-accounts|savings-accounts|credit-cards|loans[a-z/-]*)|"
    r"checking|earn-more|switch-and-earn|savings-account/[a-z-]+|personal-loans)$")


def page_kind(path: str) -> str:
    p = path.lower()
    if RATE_PAGE.match(p):
        return "rates"
    if p.startswith("calculators") or p.startswith("business-calculators") or "calculator" in p:
        return "calc"
    if p.startswith("hubfs"):
        return "feesheet" if "fee-schedule" in p else "doc"
    if p.startswith("newsroom"):
        return "news"
    if p.startswith("learn") or p.startswith("resources") or re.match(r"(19|20)\d\d/", p) or "/resources/" in p:
        return "blog"
    return "core"


# Product families -> (question trigger, authoritative url path fragments)
FAMILIES = [
    ("auto", re.compile(r"\b(auto|car|cars|vehicle|truck|suv|motorcycle|model[- ]year)\b", re.I),
     ("loans/auto-",)),
    ("heloc", re.compile(r"\b(heloc|home equity|equity line)\b", re.I), ("loans/home-equity",)),
    ("home_improvement", re.compile(r"\bhome improvement\b", re.I), ("home-improvement", "loans/home-loans")),
    ("mortgage", re.compile(r"\b(mortgage|home loan|home-loan|\d\d[- ]year fixed|\d\d[- ]year|refinanc\w*|conventional|jumbo mortgage)\b", re.I),
     ("loans/home-loans",)),
    ("personal_loan", re.compile(r"\bpersonal loans?\b", re.I), ("loans/personal-loans",)),
    ("student_loan", re.compile(r"\bstudent loans?\b", re.I), ("loans/student-loans",)),
    ("rec_loan", re.compile(r"\b(boat|rv|camper|recreational)\b", re.I), ("loans/recreational-loans",)),
    ("credit_card", re.compile(r"\b(credit cards?|visa|card apr|cards?)\b", re.I), ("credit-cards",)),
    ("ira", re.compile(r"\b(ira|iras|roth|retirement)\b", re.I), ("personal/retirement",)),
    ("certificate", re.compile(r"\b(certificates?|cds?|term[- ]share|jumbo)\b", re.I),
     ("personal/savings-account", "business-rates", "business/savings-accounts")),
    ("money_market", re.compile(r"\b(money market|achieve|investor'?s choice|high[- ]yield)\b", re.I),
     ("personal/savings-account", "business-rates", "business/savings-accounts", "earn-more")),
    ("savings", re.compile(r"\b(savings?|hsa|holiday club|launchpad|you name it|bright)\b", re.I),
     ("personal/savings-account", "business-rates", "business/savings-accounts")),
    ("checking", re.compile(r"\bchecking\b", re.I),
     ("personal/checking-accounts", "business/checking-accounts", "business/business-checking-accounts",
      "business-rates", "checking")),
    ("fee", re.compile(r"\b(fees?|charge|cost)\b", re.I), ("fee-schedule",)),
]

FAMILY_NAMES = [f[0] for f in FAMILIES]


def question_families(q: str) -> list[str]:
    fams = [name for name, rx, _ in FAMILIES if rx.search(q)]
    # "home equity" should not also fire mortgage; "certificate" beats generic savings
    if "heloc" in fams and "mortgage" in fams and not re.search(r"mortgage", q, re.I):
        fams.remove("mortgage")
    if "home_improvement" in fams and "mortgage" in fams:
        fams.remove("mortgage")
    if "certificate" in fams and "savings" in fams and not re.search(r"\bsavings? account", q, re.I):
        fams.remove("savings")
    if "money_market" in fams and "savings" in fams and not re.search(r"\bshare savings", q, re.I):
        fams.remove("savings")
    if "ira" in fams and "certificate" in fams:
        pass
    if "credit_card" in fams and "fee" in fams:
        pass
    return fams


_FAM_RX = {name: rx for name, rx, _ in FAMILIES}
_ROW_EXTRA = {
    "mortgage": re.compile(r"\b\d\d[- ]year fixed|conventional|purchase:|refinance:", re.I),
    "certificate": re.compile(r"\b\d+ month\b", re.I),
    "auto": re.compile(r"model year", re.I),
}


def row_home(fams: list, row) -> bool:
    """A row is 'official' for a question when it sits on the family's home page AND the row itself is
    about that family (the central /rates sheet and some product pages host several products, and old
    pages carry embedded rate blocks for other products)."""
    about = row.about + " " + row.section
    for f in fams:
        if not family_home(f, row.path):
            continue
        rx = _FAM_RX.get(f)
        if f == "fee" or (rx and rx.search(about)) or (f in _ROW_EXTRA and _ROW_EXTRA[f].search(about)):
            if f == "certificate" and not re.search(r"certificate|cd|term", about, re.I):
                continue
            return True
    return False


def family_home(fam: str, path: str) -> bool:
    """True when `path` is one of the official pages for product family `fam`.
    Fragments match whole path segments: "checking" matches /checking but not /checking-offer-lp;
    a fragment ending in "-" is a prefix (loans/auto- matches auto-loans and auto-motorcycle-loans)."""
    p = path.lower().strip("/")
    # /rates is the site's central rate sheet: official for every rate-bearing product
    if p == "rates" and fam not in ("fee", "student_loan", "rec_loan"):
        return True
    for name, _, frags in FAMILIES:
        if name != fam:
            continue
        for f in frags:
            if f.endswith("-"):
                if p.startswith(f) or ("/" + f) in ("/" + p):
                    return True
            elif p == f or p.endswith("/" + f) or p.startswith(f + "/"):
                return True
            elif f == "fee-schedule" and f in p:
                return True
    return False


STALE_LINK = re.compile(r"(info\.dupaco\.com|pantheonsite\.io|/connect/|/locations/|dupaco\.com/personal/rates)", re.I)
