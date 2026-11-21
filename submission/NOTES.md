# Dupaco knowledge system — notes

`submission/solution.py` → `build()` returns `DupacoKnowledge` with `ingest`, `answer`, `conflicts`.
Pure Python standard library, deterministic, **no API key needed**. Ingesting all five snapshots takes
about 5 s, and a typical answer takes under 0.1 s.

```
python harness/run_eval.py --submission submission --eval eval/public_eval.jsonl
```

## How it works

1. **Versioned store (`dupaco_kb/store.py`)**. Every snapshot becomes a *view* of the site on its date.
   Pages, table rows and text passages are deduplicated across snapshots by content hash, so five
   near-identical crawls cost about the same as one. A page that drops out of a later crawl is carried
   forward, but flagged as carried so it ranks below pages seen in that crawl. From each page the store extracts:
   - **rate/fee rows** from markdown tables (and rate sentences on product pages). Each row keeps its
     product context, column headers, term (months), model-year band, balance or loan-amount tier, and
     values (APY, dividend rate, APR, rate, $ fees);
   - **passages** (heading + ~250–900 chars) indexed with BM25;
   - **branch hours** blocks ("Lobby" / "Drive-up" followed by day lines) and publication dates.
2. **Answering (`dupaco_kb/engine.py`)**. `as_of` picks the newest snapshot taken on or before that date.
   Then the question is routed:
   - *refusal checks*: a proper name the site never mentions, a branch with no published hours, or a
     rate question whose product or term has no row (7/1 ARM, construction loan, 7-year CD) →
     `confident=False`, "isn't listed … please contact Dupaco" (no numbers);
   - *rates and fees* → score every row on: product family, qualifier match (term, model year, balance
     or loan amount, jumbo vs standard, personal vs business, IRA, purchase vs refinance), page authority,
     and word overlap. Only the winning tier is returned. List questions ("what credit cards…",
     "mortgage rates right now") return every current row of the official table. "Highest/lowest" picks the
     extreme tier of the same product. Change questions add the value history across snapshots;
   - *hours*, *addresses*, *phone*, *URLs* (stale hosts filtered out), *"how many"* counts → small
     specialised extractors;
   - everything else → BM25 passages weighted by page kind and recency.
   Every answer cites its sources (page URL + snapshot id) and sets `conflict_note` when another page
   in the same snapshot shows a different value.
3. **Conflicts (`dupaco_kb/conflicts.py`)**. Each rate row is mapped to a *topic* (for example "60-month auto loan
   APR" or "10-month jumbo certificate special APY"). Qualifiers are part of the topic, so different
   tiers never count as conflicts. A topic is reported when the site showed different values for it,
   either on two pages at once or on the official page across snapshots. The report gives the winner and
   the reason. Prose contradictions in press-release boilerplate (branch count, "Nth time Forbes…") are
   also reported.

## What counts as "current" (the rules)

1. **Date first.** Only snapshots taken on or before `as_of` (default: the newest) count, and each page
   is used in its newest version as of that date.
2. **Authority next.** Each product's own rate page and the central `/rates` sheet are official. They
   beat other first-party pages, which beat the fee PDF and other PDFs, newsroom and blog posts, and
   finally calculators. Calculators and old landing pages are known to keep stale rates (for example the
   vehicle-affordability calculator still shows 5.34%; the auto page says 5.39%). A row only counts as
   official if it is *about* that product: old pages carry embedded rate blocks for other products.
3. **Newest statement wins among equals.** Recurring press-release facts ("22 branch office locations")
   come from the most recently published release. Pages carried forward from an earlier crawl get a
   penalty.
4. **Expired promotions lose.** Rows that say "ended", "expired" or "no longer available" are demoted,
   and specials that disappeared from the rate sheet are no longer offered.
5. **Qualifiers are facts, not conflicts.** 6-month ≠ 12-month, rate ≠ APY, jumbo ≠ standard,
   personal ≠ business, purchase ≠ refinance, $100k tier ≠ $300k tier.
6. **Be honest.** If no official row or passage supports the specific product, term or branch asked about,
   the system declines instead of guessing.

## Optional LLM (off by default)

Set `KB_USE_LLM=1` **and** `ANTHROPIC_API_KEY` (and `pip install anthropic`) to let Claude
(`KB_LLM_MODEL`, default `claude-sonnet-5-5`, 8 s timeout) re-phrase the rule-based answer into friendlier
prose. The rule engine still chooses the facts. The model's output is rejected, and the rule answer
kept, if it adds, drops or changes any number or URL, errors, or times out. Without the flag or key no
network call is made. Grading needs neither; the paid API is used only if you opt in.
