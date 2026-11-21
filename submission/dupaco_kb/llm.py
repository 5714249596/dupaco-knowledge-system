"""Optional LLM polish. Off unless KB_USE_LLM=1 and ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) are set
and the `anthropic` package is installed. The rule engine always produces the answer first; the model
may only re-phrase it from the same evidence. If the model introduces any number that is not in the
rule answer, times out, or errors, the rule answer is kept."""
from __future__ import annotations

import os
import re

MODEL = os.getenv("KB_LLM_MODEL", "claude-sonnet-5-5")
TIMEOUT_S = float(os.getenv("KB_LLM_TIMEOUT", "8"))
_client = None
_disabled = False

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def enabled() -> bool:
    if _disabled:
        return False
    if os.getenv("KB_USE_LLM", "").lower() not in ("1", "true", "yes", "on"):
        return False
    return bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))


def _get_client():
    global _client, _disabled
    if _client is None:
        try:
            import anthropic  # noqa: WPS433 (optional dependency)
            _client = anthropic.Anthropic(timeout=TIMEOUT_S, max_retries=0)
        except Exception:
            _disabled = True
            return None
    return _client


def _nums(s: str) -> set:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(s or "")}


def rephrase(question: str, rule_answer: str, evidence: list) -> str | None:
    client = _get_client()
    if client is None:
        return None
    ev = "\n---\n".join(e for e in evidence[:4] if e)[:4000]
    prompt = (
        "You are a credit-union member-service assistant. Re-write the DRAFT ANSWER into one or two clear, "
        "friendly sentences for the member. Use ONLY facts from the draft answer. Do not add, drop or change any "
        "number, rate, date, URL or name that answers the question; do not mention other tiers or old values. "
        "If the draft lists several rows, keep every row's values.\n\n"
        f"QUESTION: {question}\n\nDRAFT ANSWER:\n{rule_answer}\n\nSUPPORTING TEXT (context only):\n{ev}\n\n"
        "Reply with the re-written answer only."
    )
    try:
        resp = client.messages.create(
            model=MODEL, max_tokens=400, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        out = "".join(getattr(b, "text", "") for b in resp.content).strip()
    except Exception:
        return None
    if not out:
        return None
    if not _nums(out) <= _nums(rule_answer):          # model invented a number -> reject
        return None
    if not _nums(rule_answer) <= _nums(out) and len(_nums(rule_answer)) <= 6:   # dropped a value -> reject
        return None
    for url in re.findall(r"[a-z0-9.-]+\.com[^\s)]*", rule_answer):
        if url.rstrip(".") not in out:
            return None
    return out
