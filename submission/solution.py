"""
Dupaco knowledge system — knows what is true *now* (or on any `as_of` date).

    ks = build()
    for snap in snapshots: ks.ingest(snap)        # oldest first
    ks.answer("What's the 60-month auto loan APR?")  # -> Answer
    ks.conflicts()                                 # -> list[Conflict]

Pure Python by default. Set KB_USE_LLM=1 *and* ANTHROPIC_API_KEY to let Claude re-phrase the
rule-based answer (numbers are checked against the evidence; any problem falls back to rules).
See NOTES.md.
"""
from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(next(d / "harness" for d in HERE.parents if (d / "harness" / "contract.py").exists())))

from contract import Answer, Conflict, Snapshot, Source  # noqa: E402

from dupaco_kb import llm  # noqa: E402
from dupaco_kb.conflicts import find_conflicts  # noqa: E402
from dupaco_kb.engine import Engine  # noqa: E402
from dupaco_kb.store import Store  # noqa: E402


def _norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w$%.\- ]+", " ", q.lower())).strip()


class DupacoKnowledge:
    def __init__(self):
        self.store = Store()
        self.engine = Engine(self.store)
        self._cache: dict = {}
        self._conflicts = None
        self._lock = threading.Lock()

    def ingest(self, snapshot: Snapshot) -> None:
        self.store.add_snapshot(snapshot.snapshot_id, snapshot.taken_on, snapshot.pages)
        self._cache.clear()
        self._conflicts = None

    def answer(self, question: str, as_of=None) -> Answer:
        if not self.store.views:
            return Answer(text="I don't know — no website data has been loaded yet.", confident=False)
        key = (_norm_q(question), as_of or "")
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        res = self.engine.answer(question, as_of)
        text = res.text
        if res.confident and llm.enabled():
            better = llm.rephrase(question, res.text, res.evidence)
            if better:
                text = better
        ans = Answer(
            text=text,
            sources=[Source(url=u, snapshot_id=sid, quote=qt) for u, sid, qt in res.sources],
            confident=res.confident,
            conflict_note=res.conflict_note,
        )
        with self._lock:
            self._cache[key] = ans
        return ans

    def conflicts(self) -> list[Conflict]:
        if self._conflicts is None:
            self._conflicts = [
                Conflict(topic=c["topic"], values=c["values"],
                         sources=[Source(url=u, snapshot_id=s, quote=q) for u, s, q in c["sources"]],
                         resolution=c["resolution"], reason=c["reason"])
                for c in find_conflicts(self.store)
            ]
        return list(self._conflicts)


def build():
    return DupacoKnowledge()
