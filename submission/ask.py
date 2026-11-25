"""Ask the knowledge system questions from the command line.

    python submission/ask.py                          # interactive: type questions, blank line to quit
    python submission/ask.py "What's the HELOC rate?"  # one question
    python submission/ask.py --as-of 2026-05-31 "What was the 10-month jumbo certificate rate?"
    python submission/ask.py --conflicts               # list the contradictions it found

Run from the repo root after the snapshots are unpacked (run harness/run_eval.py once, or unzip
snapshots_zip/ into snapshots/). In interactive mode, start a line with "@YYYY-MM-DD " to ask "as of" a date.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import load_snapshots  # noqa: E402

import solution  # noqa: E402


def show(ks, question: str, as_of: str | None) -> None:
    t = time.time()
    a = ks.answer(question, as_of)
    ms = (time.time() - t) * 1000
    print(f"\n{a.text}")
    if not a.confident:
        print("  (not confident: the site doesn't say)")
    for s in a.sources[:3]:
        print(f"  source: {s.url}  [{s.snapshot_id}]")
    if a.conflict_note:
        print(f"  note: {a.conflict_note}")
    print(f"  ({ms:.0f} ms{', as of ' + as_of if as_of else ''})\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--as-of", dest="as_of")
    ap.add_argument("--snapshots", type=Path, default=ROOT / "snapshots")
    ap.add_argument("--conflicts", action="store_true")
    args = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    t = time.time()
    ks = solution.build()
    snaps = load_snapshots(args.snapshots)
    if not snaps:
        sys.exit(f"No snapshots found in {args.snapshots}. Put the kit's snapshots_zip/ folder in the repo root.")
    for s in snaps:
        ks.ingest(s)
    print(f"Loaded {len(snaps)} snapshots in {time.time() - t:.1f}s "
          f"({snaps[0].taken_on} .. {snaps[-1].taken_on})")

    if args.conflicts:
        for c in ks.conflicts():
            print(f"- {c.topic}: {', '.join(c.values[:8])} -> {c.resolution}\n    {c.reason}")
        return
    if args.question:
        show(ks, " ".join(args.question), args.as_of)
        return

    print('Type a question (prefix "@2026-05-31 " for as-of). Blank line to quit.')
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        as_of = args.as_of
        if line.startswith("@"):
            as_of, _, line = line[1:].partition(" ")
        show(ks, line, as_of)


if __name__ == "__main__":
    main()
