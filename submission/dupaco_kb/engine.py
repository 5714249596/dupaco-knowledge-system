"""Question answering over the versioned store (pure Python, deterministic)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from . import text as T
from .authority import KIND_WEIGHT, STALE_LINK, family_home, question_families, row_home
from .store import Row, Store, View

REFUSAL = ("I'm sorry, that isn't listed on Dupaco's website, so I can't confirm it. "
           "Please contact Dupaco directly or visit a branch for that information.")

RATE_Q = re.compile(r"\b(rates?|apys?|aprs?|dividends?|yields?|interest|percent|earns?|earning|pays?|paying)\b|%", re.I)
FEE_Q = re.compile(r"\b(fees?|charges?|costs?|how much (?:is|does|do|will))\b", re.I)
LIST_Q = re.compile(r"\b(rates|all|list|each|every|options|what (?:\w+ )?(?:cards|accounts|certificates|loans) do|"
                    r"right now|available|offer)\b", re.I)
URL_Q = re.compile(r"\b(link|url|website|web ?site|web address|web ?page)\b|"
                   r"\b(join|apply|enroll|sign up|open|find|see|locate|look up)\b.*\bonline\b", re.I)
CARD_LIST_Q = re.compile(r"\bwhat (?:kinds? of |types? of )?(credit cards|cards|checking accounts|savings accounts|"
                         r"certificates|cds|money market accounts)\b", re.I)
HOURS_Q = re.compile(r"\bhours?\b|\bwhat time\b|\bopen on (saturday|sunday)|\bclose[sd]? (at|on) (saturday|sunday)", re.I)
LEX_SKIP = {T.stem(w) for w in """financing finance borrow borrowing get need want buy apply take tell show give list
info information detail details quote quotes change changed since before now compared compare gone start
still went increase decrease raise raised lower lowered history previous previously used""".split()}
CHANGE_Q = re.compile(r"\b(chang\w*|used to|previous\w*|before|went (?:up|down)|increas\w*|decreas\w*|history|"
                      r"raised|lowered|differ\w* from|since)\b", re.I)
NUM_WORDS = {T.stem(w) for w in """one two three four five six seven eight nine ten eleven twelve fifteen eighteen
twenty thirty forty sixty seventy eighty""".split()}
DATE_WORDS = {T.stem(w) for w in """january february march april may june july august september october november
december mid early late ago last week weekend today yesterday current currently now""".split()}
EXPIRED = re.compile(r"\b(offer|promotion|promo|special)?\s*(has )?(ended|expired|no longer (?:available|offered))\b", re.I)
PHONE_Q = re.compile(r"\b(phone number|telephone|number to call|call (?:you|dupaco)|contact number|"
                     r"customer service number|what(?:'s| is) (?:the|your) number)\b", re.I)
PHONE = re.compile(r"\(?\b\d{3}\)?[-. ]\d{3}[-.]\d{4}\b")
ADDR_Q = re.compile(r"\b(address|located|location of|where is|where's)\b", re.I)
SUPER_HI = re.compile(r"\b(highest|best|top|maximum|max|the most)\b", re.I)
SUPER_LO = re.compile(r"\b(lowest|minimum|cheapest|least)\b", re.I)
HOW_MANY = re.compile(r"\b(?:how many|(?:total )?number of|count of) ([a-z\-]+)(?: ([a-z\-]+))?", re.I)
STREET = re.compile(r"\b\d{2,6}\s+[A-Z0-9][\w.']*(?:\s+[A-Z][\w.']*){0,3}\s+"
                    r"(?:St|Rd|Ave|Ln|Dr|Blvd|Way|Ct|Pl|Hwy|Pkwy|Lane|Street|Road|Avenue|Drive|Boulevard|Circle|Cir)\b\.?")
PROPER_SKIP = set("""dupaco what whats when where which who how can could is are do does i the a an my your our
lobby drive saturday sunday monday tuesday wednesday thursday friday branch branches atm hours visa platinum rewards
business share savings checking certificate shine apple google samsung pay iowa illinois wisconsin forbes october
november december january february march april may june july august september cd cds ira roth heloc
street st avenue ave road rd drive dr lane ln boulevard blvd way court highway""".split())

COVER_SKIP = {T.stem(w) for w in """dupaco month year model much tier balance standard fixed new used purchase vehicle car
term length option highest lowest best top maximum minimum general typical earn regular buy buying amount kind type
level current today currently price account loan mid early late ago last back then time date period was
january february march april may june july august september october november december week day
financing finance borrow borrowing get need want buy apply take open opening pay paying charge offer list tell show
give info information detail details much over about rate quote quotes number share
change changed changes since before now compared compare gone up down start still did has went increase decrease
raise raised lower lowered history previous previously used new""".split()}


@dataclass
class Result:
    text: str
    sources: list = field(default_factory=list)    # (url, sid, quote)
    confident: bool = True
    conflict_note: str | None = None
    evidence: list = field(default_factory=list)   # strings given to the optional LLM
    mode: str = ""


@dataclass
class QF:
    raw: str
    toks: list
    content: list
    fams: list
    terms: set
    year: int | None
    amount: float | None
    jumbo: bool
    business: bool
    ira: bool
    refi: bool
    listq: bool
    rateq: bool
    feeq: bool
    codes: list


def features(q: str) -> QF:
    ql = q.lower()
    toks = T.query_tokens(q)
    if re.search(r"\bstill open\b|\bclos(e|ed|ing|ure)\b", ql):
        toks += ["clos", "closur"]
    content = [t for t in dict.fromkeys(T.tokens(q)) if t not in T.GENERIC and not re.fullmatch(r"[\d.%]+", t)]
    terms = T.terms_months(q)
    year = T.question_year(q)
    amts = T.money_amounts(q)
    fams = question_families(q)
    return QF(
        raw=q, toks=toks, content=content, fams=fams, terms=terms, year=year,
        amount=amts[0] if amts else None,
        jumbo="jumbo" in ql,
        business=bool(re.search(r"\b(business|commercial)\b", ql)),
        ira=bool(re.search(r"\b(ira|iras|roth|traditional|retirement)\b", ql)),
        refi=bool(re.search(r"refinanc", ql)),
        listq=bool(LIST_Q.search(q)) and not terms and year is None and not amts,
        rateq=bool(RATE_Q.search(q)) or ("certificate" in fams and not ql.startswith("how")),
        feeq=bool(FEE_Q.search(q)),
        codes=re.findall(r"\b\d+/\d+\b", q),       # e.g. "7/1" ARM
    )


def _proper_names(q: str) -> list[str]:
    """Capitalised words that are not sentence-initial and not common banking vocabulary."""
    out = []
    for m in re.finditer(r"\b([A-Z][a-zA-Z\-]{2,})\b", q):
        w = m.group(1)
        if m.start() == 0 or q[:m.start()].rstrip().endswith((".", "?", "!")):
            continue
        if w.lower() in PROPER_SKIP or T.stem(w.lower()) in T.STOP:
            continue
        out.append(w)
    return out


class Engine:
    def __init__(self, store: Store):
        self.s = store
        self._carried: set = set()

    # ------------------------------------------------------------------ public
    def answer(self, q: str, as_of: str | None) -> Result:
        view = self.s.view_at(as_of)
        qf = features(q)

        unknown = [w for w in _proper_names(q)
                   if any(self.s.df_all.get(t, 0) == 0 for t in T.tokens(w, keep_stop=True))]
        if unknown:
            return Result(REFUSAL, confident=False, mode="refuse-unknown",
                          conflict_note=f"No page mentions {', '.join(unknown)}.")

        if HOURS_Q.search(q) and not qf.rateq:
            r = self._hours_answer(q, view)
            if r:
                return r

        if PHONE_Q.search(q) and not _proper_names(q):
            r = self._phone_answer(view)
            if r:
                return r

        if ADDR_Q.search(q) and not qf.rateq:
            r = self._address_answer(q, qf, view)
            if r:
                return r

        if URL_Q.search(q):
            r = self._url_answer(q, qf, view)
            if r:
                return r

        if CARD_LIST_Q.search(q) and qf.fams and not qf.rateq:
            qf.listq = True
            r = self._row_answer(qf, view)
            if r:
                return r

        if qf.rateq or qf.feeq:
            r = self._row_answer(qf, view)
            if r:
                return r
            if qf.rateq and qf.content:
                r = self._prose_answer(qf, view, need_rate=True)
                if r:
                    return r
                return Result(REFUSAL, confident=False, mode="refuse-rate")

        r = self._prose_answer(qf, view)
        if r:
            return r
        return Result(REFUSAL, confident=False, mode="refuse-none")

    # ------------------------------------------------------------------ hours
    def _hours_answer(self, q: str, view: View) -> Result | None:
        kind = "drive" if re.search(r"drive", q, re.I) else "lobby"
        names = _proper_names(q)
        found = []      # (pub ordinal, page, schedule)
        for pg in view.pages.values():
            for k, sched in pg.hours:
                if k == kind:
                    found.append((pg.pub.toordinal() if pg.pub else 0, pg, sched))
        label = "Lobby" if kind == "lobby" else "Drive-up"
        if names:
            hits = []
            for o, pg, sched in found:
                hay = (pg.title + " " + pg.md[:3000]).lower()
                if all(n.lower() in hay for n in names):
                    hits.append((o, pg, sched))
            if not hits:
                return Result(REFUSAL, confident=False, mode="refuse-hours",
                              conflict_note=f"No published hours found for {' '.join(names)}.")
            hits.sort(key=lambda x: -x[0])
            o, pg, sched = hits[0]
            text = f"{label} hours for the {' '.join(names)} location: {sched}."
            return Result(text, [(pg.url, view.sid, sched)], evidence=[text], mode="hours")
        if not found:
            return None
        cnt = Counter(s for _, _, s in found)
        newest = {}
        for o, pg, s in found:
            if o >= newest.get(s, (-1, None))[0]:
                newest[s] = (o, pg)
        order = sorted(cnt, key=lambda s: (-cnt[s], -newest[s][0], s))[:4]
        first = order[0]
        text = f"{label} hours vary by branch. The most common schedule is {first}."
        if len(order) > 1:
            text += " Other branches use: " + " | ".join(order[1:]) + "."
        text += " Tell me which branch you're visiting and I can give its exact hours."
        srcs = [(newest[s][1].url, view.sid, s) for s in order]
        return Result(text, srcs, evidence=[text], mode="hours")

    # ------------------------------------------------------------------ phone
    def _phone_answer(self, view: View) -> Result | None:
        """Main phone number = the number most often published on official pages (contact page first)."""
        cnt, where = Counter(), {}
        for pg in view.pages.values():
            if pg.kind not in ("core", "rates"):
                continue
            for m in PHONE.finditer(pg.md):
                num = re.sub(r"\D", "", m.group(0))[-10:]
                cnt[num] += 3 if pg.path == "contact" else 1
                where.setdefault(num, pg.url)
        if not cnt:
            return None
        num = max(sorted(cnt), key=lambda k: cnt[k])
        nice = f"{num[:3]}-{num[3:6]}-{num[6:]}"
        text = f"You can reach Dupaco at {nice}."
        return Result(text, [(where[num], view.sid, nice)], evidence=[text], mode="phone")

    # ------------------------------------------------------------------ address
    def _address_answer(self, q: str, qf: QF, view: View) -> Result | None:
        names = _proper_names(q)
        if not names:
            return None
        want_branch = bool(re.search(r"\bbranch|office|location\b", q, re.I))
        best = None
        for pg in view.pages.values():
            if pg.kind not in ("core", "rates"):
                continue
            lines = pg.md.split("\n")
            for i, line in enumerate(lines):
                if not STREET.search(line):
                    continue
                if not all(n.lower() in line.lower() for n in names):
                    continue
                ctx = " ".join(lines[max(0, i - 2):i]).lower()
                score = KIND_WEIGHT[pg.kind]
                if want_branch and "dupaco branch" in ctx:
                    score += 2
                elif want_branch and "branch" in ctx:
                    score += 1
                if "atm" in ctx and want_branch:
                    score -= 1
                if best is None or score > best[0]:
                    best = (score, pg, T.strip_links(line).strip())
        if not best:
            return None
        _, pg, line = best
        text = f"The {' '.join(names)} {'branch ' if want_branch else ''}address is {line}."
        return Result(text, [(pg.url, view.sid, line)], evidence=[text], mode="address")

    # ------------------------------------------------------------------ rows
    def _score_row(self, r: Row, qf: QF, qset: list, idfs: dict, idfsum: float) -> float:
        hit = sum(idfs[t] for t in qset if t in r.toks or any(a in r.toks for a in T.SYN.get(t, ())))
        s = 3.0 * hit / idfsum if idfsum else 0.0
        if qf.fams:
            s += 2.0 if row_home(qf.fams, r) else -1.0
        s += KIND_WEIGHT.get(r.kind, 0.5)
        lab = r.label.lower()
        if qf.terms:
            if r.terms:
                s += 2.0 if (r.terms & qf.terms) else -4.0
            else:
                s -= 0.8
        if qf.year:
            if r.myr:
                s += 2.0 if r.myr[0] <= qf.year <= r.myr[1] else -4.0
            else:
                s -= 0.6
        if qf.amount is not None and r.bal:
            lo, hi = r.bal
            s += 2.0 if lo <= qf.amount <= hi + 0.999 else -3.0
        rj = "jumbo" in lab or "jumbo" in r.context.lower()
        if qf.jumbo:
            s += 1.5 if rj else -2.5
        elif rj:
            s -= 1.0
        rb = r.path.startswith("business") or "business" in lab
        if qf.business:
            s += 1.0 if rb else -1.5
        elif rb:
            s -= 0.4 if qf.listq else 1.2
        ri = bool(re.match(r"(roth|traditional)\b", lab)) or "retirement" in r.path
        if qf.ira:
            s += 1.0 if ri else -1.5
        elif ri:
            s -= 1.5
        rr = "refinance" in lab or "refinance" in r.context.lower()
        if qf.refi:
            s += 1.0 if rr else -1.5
        elif rr:
            s -= 0.8
        if r.is_line:
            s -= 0.4
        if r.ukey in self._carried:
            s -= 0.5
        if EXPIRED.search(r.label) or EXPIRED.search(r.context):
            s -= 3.0
        if qf.feeq and not qf.rateq:
            has_money = any(v.startswith("$") for _, v in r.values)
            s += 0.5 if has_money else -1.0
        return s

    def _row_answer(self, qf: QF, view: View) -> Result | None:
        self._carried = view.carried
        if "auto" in qf.fams and qf.year is None and re.search(r"\bnew (car|vehicle|truck|auto|suv)", qf.raw, re.I):
            ends = [r.myr[1] for r in view.rows if r.myr and row_home(["auto"], r)]
            if ends:                    # "new car" = the newest model-year band on the rate sheet
                qf.year = max(ends)
        if qf.codes:
            return None
        # original question words only; synonyms count as alternative matches, never as extra demands
        qset = list(dict.fromkeys(t for t in T.tokens(qf.raw)
                                  if not re.fullmatch(r"[\d.%]+", t) and t not in DATE_WORDS
                                  and t not in LEX_SKIP and t not in NUM_WORDS))
        if not qset:
            qset = [t for t in T.tokens(qf.raw) if not re.fullmatch(r"[\d.%]+", t)]
        idfs = {t: self.s.idf(t) for t in qset}
        idfsum = sum(idfs.values())
        scored = []
        for r in view.rows:
            if r.kind == "calc" and not qf.fams:
                continue
            scored.append((self._score_row(r, qf, qset, idfs, idfsum), r))
        if not scored:
            return None
        scored.sort(key=lambda x: (-x[0], x[1].url, x[1].label))
        best_s, best = scored[0]
        if best_s < 3.2:
            return None
        if not self._covered(qf, best):
            return None
        if qf.terms and not (best.terms & qf.terms):
            return None          # e.g. a 7-year certificate: no row for that term exists

        margin = 0.75 if qf.listq else 0.35
        limit = 10 if qf.listq else 4
        chosen, seen = [], set()
        home = row_home(qf.fams, best)
        for s, r in scored:
            if s < best_s - margin:
                break
            if home and not row_home(qf.fams, r):
                continue
            k = r.label_key
            if k in seen:
                continue
            seen.add(k)
            chosen.append((s, r))
            if len(chosen) >= limit:
                break

        if SUPER_HI.search(qf.raw) or SUPER_LO.search(qf.raw):
            prod = best.label.split("|")[0].strip().lower()
            pool = [x for x in scored[:80] if x[0] >= best_s - 1.5 and not x[1].is_line
                    and x[1].label.split("|")[0].strip().lower() == prod
                    and (not home or row_home(qf.fams, x[1]))]

            def top_val(r):
                vals = [float(re.sub(r"[^\d.]", "", v)) for _, v in r.values if "%" in v and re.search(r"\d", v)]
                return max(vals) if vals else -1

            if pool:
                pick = (max if SUPER_HI.search(qf.raw) else min)(pool, key=lambda x: top_val(x[1]))
                chosen = [pick]

        # "$X or less" loan-amount tiers: a $250k loan belongs to the smallest cap that covers it
        if qf.amount is not None:
            caps = [r.bal[1] for _, r in chosen if r.bal and r.bal[0] == 0 and r.bal[1] >= qf.amount]
            if caps:
                cap = min(caps)
                chosen = [x for x in chosen if not (x[1].bal and x[1].bal[0] == 0 and x[1].bal[1] != cap)]

        note = self._row_conflict_note([r for _, r in chosen], view)
        lines = [r.render() for _, r in chosen]
        text = "\n".join(lines)
        if CHANGE_Q.search(qf.raw) and chosen:
            for _, r in chosen:
                hist = self._history(r, view)
                if hist:
                    text += "\n" + hist
                    break
        srcs = []
        for _, r in chosen:
            same = [r] + [o for o in view.rows if o is not r and o.label_key == r.label_key
                          and o.values == r.values and (not home or row_home(qf.fams, o))]
            for o in same:
                if o.url not in [u for u, _, _ in srcs]:
                    srcs.append((o.url, view.sid, r.render()[:200]))
        return Result(text=text, sources=srcs[:6], conflict_note=note, evidence=lines, mode="rows")

    def _history(self, row: Row, view: View) -> str | None:
        """Timeline of the row's headline value on the same page across snapshots up to `view`."""
        def headline(r):
            for want in ("APY", "APR", "rate", ""):
                for h, v in r.values:
                    if h == want and "%" in v:
                        return v
            return None
        seq = []
        for v in self.s.views:
            if v.taken > view.taken:
                break
            match = next((o for o in v.rows if o.ukey == row.ukey and o.label_key == row.label_key), None)
            val = headline(match) if match else None
            if val and (not seq or seq[-1][1] != val):
                seq.append((v.taken, val))
        if len(seq) < 2:
            return None
        steps = " -> ".join(f"{val} (as of {d.strftime('%b')} {d.day}, {d.year})" for d, val in seq)
        return f"History on {T.url_path(row.url) or 'home page'}: {steps}. The latest value is current."

    def _covered(self, qf: QF, r: Row) -> bool:
        weights, hit = 0.0, 0.0
        for t in qf.content:
            if t in COVER_SKIP:
                continue
            w = self.s.idf(t)
            weights += w
            alts = [t] + T.SYN.get(t, [])
            if any(a in r.toks for a in alts):
                hit += w
        return weights == 0 or hit / weights >= 0.7

    def _row_conflict_note(self, rows: list, view: View) -> str | None:
        notes = []
        for r in rows:
            mine = {v for _, v in r.values if "%" in v}
            for o in view.rows:
                if o is r or o.url == r.url or o.label_key != r.label_key or o.is_line:
                    continue
                other = {v for _, v in o.values if "%" in v}
                if other and mine and other != mine:
                    notes.append(f"{T.url_path(o.url) or 'home'} shows {', '.join(sorted(other))} for '{r.label}', "
                                 f"but the product's own rate page ({T.url_path(r.url)}) is treated as authoritative.")
                    break
        return " ".join(notes[:2]) or None

    # ------------------------------------------------------------------ prose
    def _passage_scores(self, qf: QF, view: View) -> list:
        raw = self.s.bm25(view, qf.toks)
        out = []
        for pid, sc in raw.items():
            url = self.s.p_url[pid]
            pg = view.pages.get(T.url_key(url))
            if pg is None:
                continue
            w = PROSE_WEIGHT.get(pg.kind, 0.7)
            if pg.key in view.carried:
                w *= 0.8
            if pg.pub:
                age = max(0.0, (view.taken - pg.pub).days / 365.0)
                w *= 1.0 / (1.0 + 0.1 * age)
            if qf.fams and any(family_home(f, pg.path) for f in qf.fams):
                w *= 1.25
            n = len(self.s.p_text[pid])
            if n < 80:
                w *= 0.5
            out.append((sc * w, pid, pg))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def _prose_answer(self, qf: QF, view: View, need_rate: bool = False) -> Result | None:
        ranked = self._passage_scores(qf, view)
        if not ranked:
            return None
        hm = HOW_MANY.search(qf.raw)
        if hm:
            noun = T.stem(hm.group(1).lower())
            nrx = re.escape(noun[:-1] if noun.endswith("y") else noun) + r"\w*"
            num = r"(?<![\d,.$])\b(?!(?:19|20)\d\d\b)\d{1,3}(?:,\d{3})*\+?"
            rx = re.compile(num + r"\s+(?:[a-z\-]+\s+){0,2}(?i:" + nrx + r")"
                            r"|(?i:" + nrx + r")\s+(?:\w+\s+){0,4}(?:to|of|totals?|is|reached)\s+(?:more than\s+|over\s+|nearly\s+)?"
                            + num)
            about_us = re.compile(r"\b(dupaco|it has|we have|we've|our|it is|credit union)\b", re.I)

            elsewhere = re.compile(r"U\.S\.|nationwide|national|across the country|united states|industry", re.I)

            def counts_us(pid):
                body = T.strip_links(self.s.p_text[pid])
                if not about_us.search(body):
                    return False
                for sent in re.split(r"(?<=[.!?])\s+|\n", body):
                    if rx.search(sent) and not elsewhere.search(sent):
                        return True
                return False

            filt = [x for x in ranked[:400] if counts_us(x[1])]
            if filt:
                # several pages state the count; the most recently published statement is current
                top = filt[0][0]
                filt = [x for x in filt if x[0] >= 0.3 * top]
                filt.sort(key=lambda x: (-(x[2].pub.toordinal() if x[2].pub else 0), -x[0]))
                ranked = filt
        if need_rate:
            keep = []
            for x in ranked[:60]:
                body = T.strip_links(self.s.p_head[x[1]] + " " + self.s.p_text[x[1]])
                if not re.search(r"\d\.\d+\s?%", body):
                    continue
                if qf.codes and not all(c in body for c in qf.codes):
                    continue
                if qf.terms and not (T.terms_months(body) & qf.terms):
                    continue
                btoks = set(T.tokens(body))
                need = [t for t in qf.content if t not in COVER_SKIP][:4]
                if all(any(a in btoks for a in [t] + T.SYN.get(t, [])) for t in need):
                    keep.append(x)
            if not keep:
                return None
            ranked = keep
        if re.search(r"\bhow much\b|\bup to\b|\bamount\b", qf.raw, re.I) and not hm:
            top = ranked[0][0]
            money = [x for x in ranked[:10] if x[0] >= 0.6 * top and T.MONEYV.search(self.s.p_text[x[1]])]
            if money:
                ranked = money + [x for x in ranked if x not in money]
        sc, pid, pg = ranked[0]
        ptoks = set(T.tokens(self.s.p_head[pid] + " " + self.s.p_text[pid]))
        cont = [t for t in qf.content if t not in COVER_SKIP]
        if cont:
            cov = sum(self.s.idf(t) for t in cont if t in ptoks or any(a in ptoks for a in T.SYN.get(t, [])))
            tot = sum(self.s.idf(t) for t in cont)
            if tot and cov / tot < 0.1:
                return None
        text = self._passage_text(pid)
        if hm and ranked is not None:
            body = T.strip_links(self.s.p_text[pid])
            sents = [s for s in re.split(r"(?<=[.!?])\s+|\n", body) if rx.search(s)]
            if sents:
                text = " ".join(sents[:2]).strip()
        used = [pid]
        # add the runner-up passage when it is nearly as good (helps list-style answers)
        for sc2, pid2, pg2 in ranked[1:4]:
            if sc2 >= 0.8 * sc and len(text) < 500 and not hm:
                t2 = self._passage_text(pid2)
                if t2[:80] not in text:
                    text = text + "\n" + t2
                    used.append(pid2)
                break
        if len(text) > 1000:
            text = text[:1000].rsplit(" ", 1)[0] + "…"
        srcs = []
        for _, p2, pg2 in ranked[:8]:
            if pg2.url not in [s[0] for s in srcs]:
                srcs.append((pg2.url, view.sid, T.strip_links(self.s.p_text[p2])[:160] if p2 in used else None))
            if len(srcs) >= 5:
                break
        ev = [text] + [T.strip_links(self.s.p_text[p])[:600] for _, p, _ in ranked[1:3]]
        return Result(text=text, sources=srcs, evidence=ev, mode="prose")

    def _passage_text(self, pid: int) -> str:
        body = T.strip_links(self.s.p_text[pid])
        head = self.s.p_head[pid]
        text = body if (not head or head.lower() in body.lower()[:200]) else f"{head}: {body}"
        return re.sub(r"\n{2,}", "\n", text).strip()

    # ------------------------------------------------------------------ urls
    def _url_answer(self, q: str, qf: QF, view: View) -> Result | None:
        want = [t for t in dict.fromkeys(qf.toks)
                if t not in {"link", "url", "websit", "web", "site", "address", "page", "see", "use", "go", "visit"}]
        if not want:
            return None
        wset = set(want)
        cands: dict = {}
        for pg in view.pages.values():
            if pg.kind not in ("core", "rates"):
                continue
            for anchor, url in T.links(pg.md):
                ul = url.lower()
                if not ul.startswith("http") or "dupaco.com" not in ul or STALE_LINK.search(ul):
                    continue
                if any(x in ul for x in ("/hubfs/", "/hs-fs/", "mailto:", "tel:", "/newsroom/", "/learn/")):
                    continue
                key = url.split("?")[0].rstrip("/")
                utoks = set(T.tokens(re.sub(r"[/#._\-:]", " ", T.url_key(key).replace("dupaco com", ""))))
                atoks = set(T.tokens(anchor))
                ov = sum(self.s.idf(t) for t in wset & (utoks | atoks))
                if ov <= 0:
                    continue
                path = T.url_path(key)
                bonus = 2.0 if qf.fams and any(family_home(f, path.split("#")[0]) for f in qf.fams) else 0.0
                if "#rates" in key.lower() and qf.rateq:
                    bonus += 1.0
                c = cands.setdefault(key, [0.0, 0])
                c[0] = max(c[0], ov + bonus + 0.5 * len(wset & utoks))
                c[1] += 1
        if not cands:
            return None
        def depth(k):
            return T.url_key(k).split("#")[0].count("/")

        best = max(sorted(cands), key=lambda k: (cands[k][0] + 0.3 * min(cands[k][1], 20) ** 0.5 - 0.4 * depth(k)))
        text = f"You can use {best}"
        return Result(text=text, sources=[(best, view.sid, None)], evidence=[text], mode="url")


PROSE_WEIGHT = {"rates": 1.0, "core": 0.95, "feesheet": 0.9, "news": 0.85, "blog": 0.7, "doc": 0.5, "calc": 0.35, "other": 0.7}
