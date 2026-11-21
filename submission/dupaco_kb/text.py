"""Text helpers: cleaning, tokenising, date parsing and qualifier extraction."""
from __future__ import annotations

import html
import re
from datetime import date

# --------------------------------------------------------------------------- cleaning

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]*)[^)]*\)")
_FOOT = re.compile(r"(\d\.\d+%|\d%)(\d{1,2})(?![\d%]|\.\d)")   # "4.20%1" -> "4.20%"
_FOOT2 = re.compile(r"\b(APR|APY)\d\b")


def clean_markdown(md: str) -> str:
    md = md.replace("�", "-").replace("‑", "-").replace(" ", " ")
    md = md.replace("\\r\\n", " ").replace("\\\\", " ")
    if "&" in md:
        md = html.unescape(html.unescape(md))
    md = _IMG.sub("", md)
    md = _FOOT.sub(r"\1", md)
    md = _FOOT2.sub(r"\1", md)
    return md


def strip_links(text: str) -> str:
    """'[anchor](url)' -> 'anchor'; also drop bare markdown emphasis."""
    text = _LINK.sub(lambda m: m.group(1), text)
    text = text.replace("**", "").replace("\\*", "*").replace("\\#", "#")
    return re.sub(r"[ \t]+", " ", text).strip()


def links(text: str) -> list[tuple[str, str]]:
    return [(m.group(1), m.group(2)) for m in _LINK.finditer(text)]


def norm_dash(s: str) -> str:
    return s.replace("–", "-").replace("—", "-").replace("−", "-")


# --------------------------------------------------------------------------- urls

def url_key(url: str) -> str:
    u = url.strip().lower()
    u = re.sub(r"^https?://", "", u)
    u = u.removeprefix("www.")
    u = u.split("#")[0]
    u = re.sub(r"\?hslang=[a-z-]+$", "", u)
    return u.rstrip("/")


def url_path(url: str) -> str:
    k = url_key(url)
    return k.split("/", 1)[1] if "/" in k else ""


# --------------------------------------------------------------------------- tokens

STOP = set("""a an the and or of to in on for at by with from as is are was were be been being do does did
i me my we our you your it its this that these those there here what whats which who whom whose when where why how
can could would should will shall may might must am im ive id youre dont doesnt please tell give get let
any some all about into out up down over under than then so if not no yes just also very more most much many
have has had having know like want need looking look see show provide find current currently right now today
dupaco dupacos credit union cu offer offers offered between their they them theirs us ours one ones
something anything thing things kind sort way ways s t ll re ve d m""".split())

GENERIC = STOP | set("""rate rates apr apy interest dividend yield annual percentage what account accounts loan loans
current much cost costs charge charged pay paid earn earning get option options product products available program""".split())

_TOK = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?|\$")


_SUFFIXES = ("izations", "ization", "itions", "ition", "ations", "ation", "nesses", "ness", "ings", "ing",
             "ized", "izes", "ize", "edly", "ed", "ly")


def stem(t: str) -> str:
    if not t or t[0].isdigit():
        return t
    if len(t) > 4 and t.endswith("ies"):
        t = t[:-3] + "y"
    elif len(t) > 4 and t.endswith(("ches", "shes", "xes", "sses")):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    for suf in _SUFFIXES:
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            t = t[: -len(suf)]
            break
    if len(t) > 4 and t.endswith("e"):
        t = t[:-1]
    return t


SYN = {
    "cd": ["certificate"], "cds": ["certificate"], "certificate": ["certificate"],
    "mortgage": ["mortgage", "conventional", "home"], "heloc": ["heloc", "equity", "line"],
    "car": ["auto", "vehicle"], "vehicle": ["auto", "vehicle"], "auto": ["auto", "vehicle"],
    "truck": ["auto", "vehicle"], "motorcycle": ["motorcycle", "auto"],
    "apy": ["apy", "yield"], "apr": ["apr"], "ira": ["ira", "retirement", "roth", "traditional"],
    "card": ["card", "visa"], "visa": ["visa", "card"], "cc": ["card", "visa"],
    "hsa": ["health", "saving"], "kid": ["kid", "dopack"], "teen": ["launchpad", "teen"],
    "phone": ["phone", "call"], "telephone": ["phone", "call"],
}


STOP = STOP | {stem(w) for w in STOP}
GENERIC = GENERIC | {stem(w) for w in GENERIC}
SYN = {stem(k): [stem(x) for x in v] for k, v in SYN.items()}


def tokens(text: str, keep_stop: bool = False) -> list[str]:
    out = []
    for t in _TOK.findall(norm_dash(text.lower())):
        if t == "$":
            continue
        t = stem(t)
        if not keep_stop and t in STOP:
            continue
        out.append(t)
    return out


def query_tokens(text: str) -> list[str]:
    out = []
    for t in tokens(text):
        out.append(t)
        for s in SYN.get(t, []):
            if s != t:
                out.append(s)
    return out


# --------------------------------------------------------------------------- dates

MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}
_MDY = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)"
                  r"\s+(\d{1,2}),\s+(20\d\d|19\d\d)\b")


def parse_iso(s: str | None) -> date | None:
    if not s:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def page_pub_date(url: str, md: str) -> date | None:
    m = re.search(r"/((?:19|20)\d\d)/(\d\d)/(\d\d)/", url)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    head = md[:600]
    m = _MDY.search(head)
    if m:
        try:
            return date(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------- qualifiers

_TERM = re.compile(r"\b(\d{1,3})\s*[- ]?\s*(months?\b|mos?\.?(?![a-z])|years?\b|yrs?\b)(?![- ]?old)", re.I)
_YEARRANGE = re.compile(r"\b((?:19|20)\d\d)\s*-\s*(\d{2,4})\b")
_YEAR = re.compile(r"\b((?:19|20)\d\d)\b")
_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(k|thousand|million|m)?\b", re.I)
_RANGE = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*(?:-|to)\s*\$?\s?([\d,]+(?:\.\d+)?)")
_PLUS = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*\+")
_ORLESS = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*or less", re.I)


def _f(s: str) -> float:
    return float(s.replace(",", ""))


_WORDNUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "eighteen": 18, "twenty": 20, "thirty": 30,
            "twenty-four": 24, "thirty-six": 36, "forty-eight": 48, "sixty": 60, "seventy-two": 72, "eighty-four": 84}
_WORDNUM_RX = re.compile(r"\b(" + "|".join(sorted(_WORDNUM, key=len, reverse=True)) + r")(?=[\s-]+(?:months?|years?|yrs?)\b)", re.I)


def terms_months(text: str) -> set[int]:
    out = set()
    text = _WORDNUM_RX.sub(lambda m: str(_WORDNUM[m.group(1).lower()]), text)
    for m in _TERM.finditer(norm_dash(text)):
        n = int(m.group(1))
        unit = m.group(2).lower()
        out.add(n * 12 if unit.startswith("y") else n)
    return out


def model_year_range(text: str) -> tuple[int, int] | None:
    t = norm_dash(text)
    if "model year" not in t.lower() and "model-year" not in t.lower():
        return None
    m = _YEARRANGE.search(t)
    if m:
        a = int(m.group(1))
        b = m.group(2)
        b = int(b) if len(b) == 4 else int(str(a)[:2] + b)
        return (min(a, b), max(a, b))
    m = _YEAR.search(t)
    if m:
        y = int(m.group(1))
        if "older" in t.lower() or "and prior" in t.lower():
            return (1900, y)
        return (y, y)
    return None


def question_year(text: str) -> int | None:
    t = text.lower()
    if not re.search(r"model|vehicle|car|auto|truck|motorcycle|suv", t):
        return None
    m = _YEAR.search(text)
    return int(m.group(1)) if m else None


def money_amounts(text: str) -> list[float]:
    out = []
    for m in _MONEY.finditer(text):
        v = _f(m.group(1))
        suf = (m.group(2) or "").lower()
        if suf in ("k", "thousand"):
            v *= 1000
        elif suf in ("m", "million"):
            v *= 1_000_000
        out.append(v)
    return out


def balance_range(text: str) -> tuple[float, float] | None:
    t = norm_dash(text)
    m = _RANGE.search(t)
    if m:
        return (_f(m.group(1)), _f(m.group(2)))
    m = _PLUS.search(t)
    if m:
        return (_f(m.group(1)), float("inf"))
    m = _ORLESS.search(t)
    if m:
        return (0.0, _f(m.group(1)))
    return None


PCT = re.compile(r"\d+(?:\.\d+)?\s?%")
MONEYV = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
