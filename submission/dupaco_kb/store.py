"""Versioned page store: every snapshot is kept as a *view*; pages, table rows and text passages
are deduplicated across snapshots so five near-identical crawls cost little more than one."""
from __future__ import annotations

import hashlib
import math
import re
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from datetime import date

from . import text as T
from .authority import KIND_WEIGHT, page_kind


@dataclass
class Row:
    url: str
    path: str
    kind: str
    label: str                 # non-value cells joined
    context: str               # table caption / nearest heading
    header: list[str]
    values: list[tuple[str, str]]   # (column header, value) for % / $ cells
    toks: frozenset
    terms: frozenset
    myr: tuple | None
    bal: tuple | None
    is_line: bool = False      # came from a prose line instead of a table
    section: str = ""          # product words found just above the table

    @cached_property
    def about(self) -> str:
        """Text describing what product this row is about (for family checks)."""
        return f"{self.context} {self.label} {' '.join(self.header)}"

    @cached_property
    def key(self) -> str:
        return re.sub(r"[^a-z0-9$%+.]+", " ", T.norm_dash((self.context + " " + self.label).lower())).strip()

    @cached_property
    def label_key(self) -> str:
        s = T.norm_dash(self.label.lower()).replace(",", "")
        s = re.sub(r"\.00\b", "", s)
        return re.sub(r"[^a-z0-9$%+.]+", " ", s).strip()

    def render(self) -> str:
        lab = self.label.strip(" |")
        if self.is_line:
            body = re.sub(r"\s+\d\.?$", "", T.strip_links(self.label).lstrip("-* ").strip())
            ctx = self.context.strip()
            if ctx and len(ctx) < 60 and ctx.lower() not in body.lower():
                return f"{ctx}: {body}"
            return body
        vals = ", ".join(f"{h} {v}" if h and h.lower() not in v.lower() else v for h, v in self.values)
        ctx = self.context.strip()
        if ctx and ctx.lower() not in lab.lower():
            lab = f"{ctx}: {lab}" if lab else ctx
        return f"{lab} — {vals}" if vals else lab


@dataclass
class Page:
    url: str
    key: str
    path: str
    title: str
    md: str
    kind: str
    weight: float
    pub: date | None
    rows: list[Row] = field(default_factory=list)
    pids: list[int] = field(default_factory=list)
    hours: list = field(default_factory=list)     # [(kind 'lobby'|'drive', schedule text)]


@dataclass
class View:
    sid: str
    taken: date
    pages: dict           # url_key -> Page
    pids: set             # active passage ids
    rows: list            # list[Row]
    carried: set = field(default_factory=set)   # url keys carried forward from an earlier pull


_SEP = re.compile(r"^\|?\s*:?-{2,}")
_HEAD = re.compile(r"^#{1,6}\s+(.*)")
_VAL = re.compile(r"\d(?:\.\d+)?\s?%|\$\s?\d")


def _short_hdr(h: str) -> str:
    hl = h.lower()
    if "annual percentage yield" in hl or hl.strip().startswith("apy"):
        return "APY"
    if "dividend" in hl:
        return "dividend rate"
    if "apr" in hl or "annual percentage rate" in hl:
        return "APR"
    if hl.strip() in ("rate", "interest rate"):
        return "rate"
    if hl.startswith("col"):
        return ""
    return h.strip()


def parse_rows(page_url: str, path: str, kind: str, title: str, md: str) -> list[Row]:
    rows: list[Row] = []
    lines = md.split("\n")
    heading = ""
    last_text = ""
    recent: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        hm = _HEAD.match(line)
        if hm:
            heading = T.strip_links(hm.group(1))
            last_text = heading
            recent.append(heading)
            i += 1
            continue
        if line.startswith("|"):
            block = []
            while i < n and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            header = [c.strip() for c in block[0].strip("|").split("|")]
            body = [b for b in block[1:] if not _SEP.match(b)]
            ctx = last_text if len(last_text) < 90 else heading
            # product words from the few lines above the table (e.g. a "Regular" table that follows
            # certificate footnotes) so the rows still know what product they describe
            section = ""
            if not _PRODUCT_WORD.search(ctx + " " + block[0]):
                section = " ".join(sorted({m.group(0).lower() for m in _PRODUCT_WORD.finditer(" ".join(recent[-6:]))}))
            for b in body:
                cells = [c.strip() for c in b.strip("|").split("|")]
                vals, lab = [], []
                for j, c in enumerate(cells):
                    hdr = header[j] if j < len(header) else ""
                    if PCT_CELL.search(c) and len(c) < 40:
                        vals.append((_short_hdr(hdr), c))
                    elif MONEY_CELL.match(c) and j > 0 and not RANGE_CELL.search(c[:40]):
                        h = _short_hdr(hdr)
                        if "payment" in h.lower():
                            continue
                        if len(c) > 60:          # "$30.00 per item *You will not be charged ..." -> keep the price
                            c = re.split(r"\s\*|\.\s", c)[0][:60]
                        vals.append((h, c))
                    elif c and c not in ("—", "-", "–", "None") and not re.fullmatch(r"[\d.]+", c):
                        lab.append(T.strip_links(c))
                if not vals:
                    continue
                label = " | ".join(lab)
                full = f"{title} {path} {ctx} {section} {label} {' '.join(header)}"
                rows.append(Row(page_url, path, kind, label, ctx, header, vals,
                                frozenset(T.tokens(full)), frozenset(T.terms_months(label)),
                                T.model_year_range(label + " " + ctx), T.balance_range(label),
                                section=section))
            continue
        if line:
            plain = T.strip_links(line)
            if kind in ("rates", "core") and PCT_CELL.search(plain) and len(plain) < 420 \
                    and re.search(r"apr|apy|rate|dividend", plain, re.I) \
                    and not re.search(r"example|illustrative|if you borrow|for instance", plain, re.I):
                full = f"{title} {path} {heading} {plain}"
                vals = [("", v) for v in T.PCT.findall(plain)]
                rows.append(Row(page_url, path, kind, plain, heading, [], vals,
                                frozenset(T.tokens(full)), frozenset(T.terms_months(plain)),
                                T.model_year_range(plain), None, is_line=True))
            if len(plain) > 2:
                last_text = plain
                recent.append(plain[:300])
        i += 1
    return rows


_PRODUCT_WORD = re.compile(r"\b(certificates?|jumbo|savings|checking|money market|ira|roth|auto|mortgage|heloc|"
                           r"home equity|visa|credit card|personal loan|business|boat|rv|recreational|motorcycle|"
                           r"loans?|heloc|health savings|hsa)\b", re.I)


_DAY = re.compile(r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)\b", re.I)
_HLABEL = re.compile(r"^\**\s*(lobby|drive[- ]?up|drive[- ]?thru|drive[- ]through)\s*(?:hours)?\s*:?\**\s*:?\s*(.*)$", re.I)


def parse_hours(md: str) -> list:
    """Find 'Lobby' / 'Drive-up' blocks followed by day lines; return normalised schedules."""
    out = []
    lines = [l.strip() for l in md.split("\n")]
    i = 0
    while i < len(lines):
        m = _HLABEL.match(lines[i])
        if m:
            kind = "lobby" if m.group(1).lower().startswith("lobby") else "drive"
            days = []
            rest = m.group(2).strip()
            if rest and _DAY.match(rest):
                days.append(rest)
            j = i + 1
            while j < len(lines) and len(days) < 6:
                l = lines[j]
                if not l:
                    j += 1
                    continue
                if _DAY.match(l):
                    days.append(l)
                    j += 1
                    continue
                break
            if days:
                sched = "; ".join(T.norm_dash(d).replace("–", "-") for d in days)
                out.append((kind, sched))
                i = j
                continue
        i += 1
    return out


RANGE_CELL = re.compile(r"\$[\d,.]+\s*(?:[-–—]|to)\s*\$?[\d,.]+|\$[\d,.]+\s*\+|or less|or more", re.I)
PCT_CELL = re.compile(r"\d(?:\.\d+)?\s?%")
MONEY_CELL = re.compile(r"^\$\s?\d[\d,]*(?:\.\d+)?\b")


def split_passages(md: str) -> list[tuple[str, str]]:
    """Return (heading, text) passages of ~250-900 chars; tables become one passage per row."""
    out = []
    heading = ""
    buf: list[str] = []
    size = 0

    def flush():
        nonlocal buf, size
        if buf:
            out.append((heading, "\n".join(buf)))
        buf, size = [], 0

    lines = md.split("\n")
    header = None
    for raw in lines:
        line = raw.strip()
        if not line:
            if size >= 250:
                flush()
            continue
        hm = _HEAD.match(line)
        if hm:
            flush()
            heading = T.strip_links(hm.group(1))[:160]
            header = None
            continue
        if line.startswith("|"):
            if _SEP.match(line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if header is None:
                header = cells
                continue
            flush()
            txt = "; ".join(f"{h}: {c}" if h and not h.lower().startswith("col") else c
                            for h, c in zip(header + [""] * len(cells), cells) if c)
            out.append((heading, txt))
            continue
        header = None
        if line.startswith("[") and line.endswith(")") and line.count("](") == 1 and len(line) < 200:
            # stand-alone button links: keep them (URL questions) but don't let them dominate
            pass
        buf.append(line)
        size += len(line)
        if size >= 900:
            flush()
    flush()
    return out


class Store:
    def __init__(self):
        self.views: list[View] = []
        self._page_cache: dict = {}       # (url_key, md hash) -> Page (without snapshot info)
        self._pid_by_key: dict = {}       # (url_key, text hash) -> pid
        self.p_text: list[str] = []
        self.p_head: list[str] = []
        self.p_url: list[str] = []
        self.p_len: array = array("I")
        self.postings: dict = defaultdict(lambda: (array("I"), array("H")))
        self.df_all: Counter = Counter()
        self._avg = None

    # ------------------------------------------------------------------ ingest
    def add_snapshot(self, sid: str, taken_on: str, pages) -> View:
        taken = T.parse_iso(taken_on) or date.today()
        vpages, pids, rows = {}, set(), []
        for p in pages:
            key = T.url_key(p.url)
            h = hashlib.md5(p.markdown.encode("utf-8", "replace")).hexdigest()
            pg = self._page_cache.get((key, h))
            if pg is None:
                md = T.clean_markdown(p.markdown)
                path = T.url_path(p.url)
                kind = page_kind(path)
                title = (p.title or "").split("|")[0].strip()
                pg = Page(p.url, key, path, title, md, kind, KIND_WEIGHT[kind], T.page_pub_date(p.url, md))
                pg.rows = parse_rows(p.url, path, kind, title, md)
                for r in pg.rows:
                    r.ukey = key
                pg.hours = parse_hours(md)
                for head, txt in split_passages(md):
                    pg.pids.append(self._add_passage(key, p.url, head or title, txt))
                self._page_cache[(key, h)] = pg
            # same URL appearing twice in one crawl (e.g. ?hsLang=en): keep the first
            if key in vpages:
                continue
            vpages[key] = pg
            pids.update(pg.pids)
            rows.extend(pg.rows)
        # Pages that were crawled before but are missing from this pull are carried forward (a crawl
        # can simply miss a page); they are flagged so they rank below pages seen in this pull.
        carried = set()
        earlier = [v for v in self.views if v.taken <= taken]
        if earlier:
            prev = max(earlier, key=lambda v: v.taken)
            for key, pg in prev.pages.items():
                if key not in vpages:
                    vpages[key] = pg
                    pids.update(pg.pids)
                    rows.extend(pg.rows)
                    carried.add(key)
        v = View(sid, taken, vpages, pids, rows, carried)
        self.views.append(v)
        self.views.sort(key=lambda x: x.taken)
        return v

    def _add_passage(self, key: str, url: str, head: str, txt: str) -> int:
        hk = (key, hashlib.md5((head + "\x00" + txt).encode()).hexdigest())
        pid = self._pid_by_key.get(hk)
        if pid is not None:
            return pid
        pid = len(self.p_text)
        self._avg = None
        self._pid_by_key[hk] = pid
        self.p_text.append(txt)
        self.p_head.append(head)
        self.p_url.append(url)
        toks = T.tokens(head + " " + T.strip_links(txt))
        self.p_len.append(len(toks))
        for t, c in Counter(toks).items():
            ids, tfs = self.postings[t]
            ids.append(pid)
            tfs.append(min(c, 65535))
            self.df_all[t] += 1
        return pid

    # ------------------------------------------------------------------ lookup
    def view_at(self, as_of: str | None) -> View:
        if not as_of:
            return self.views[-1]
        d = T.parse_iso(as_of)
        if d is None:
            return self.views[-1]
        best = self.views[0]
        for v in self.views:
            if v.taken <= d:
                best = v
        return best

    def bm25(self, view: View, qtoks: list[str], k1: float = 1.2, b: float = 0.6) -> dict:
        N = max(1, len(self.p_text))
        if self._avg is None:
            self._avg = (sum(self.p_len) / N) if N else 1.0
        avg = self._avg
        scores: dict = defaultdict(float)
        active = view.pids
        for t in set(qtoks):
            if t not in self.postings:
                continue
            ids, tfs = self.postings[t]
            df = len(ids)
            if df > N * 0.25:
                continue
            idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
            w = idf * (2.0 if qtoks.count(t) > 1 else 1.0)
            for pid, tf in zip(ids, tfs):
                if pid in active:
                    L = self.p_len[pid]
                    scores[pid] += w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * L / avg))
        return scores

    def idf(self, t: str) -> float:
        N = max(1, len(self.p_text))
        df = self.df_all.get(t, 0)
        return math.log(1 + (N - df + 0.5) / (df + 0.5))
