# When the AI Rewrites Its Own History

Compaction hallucinations, measured across 337 audited summaries.

**Johnny Morgan**, University of Maryland, Baltimore County

Companion to [`../hallucination-taxonomy`](../hallucination-taxonomy), which
covers all eight classes. This one goes deep on a single class.

| File | What |
|---|---|
| `main.pdf` | The paper |
| `main.tex` | LaTeX source |
| `Compaction_Hallucinations_Briefing.pptx` | 5 slides plus one marked backup, speaker notes throughout |
| `figures/` | 4 figures at 200 dpi |
| `figures/*.csv` | Every number on every slide |
| `make_figures.py`, `make_deck.py` | Deterministic regeneration |
| [`../figkit.py`](../figkit.py), [`../deckkit.py`](../deckkit.py) | Shared palette and layout primitives |

## What compaction is

Every model has a fixed working memory. A long session fills it. When it fills,
the model writes a summary of the session so far, the original messages are
dropped, and work continues from the summary.

This is not the model forgetting something you told it. It is the model rewriting
the record of its own work and then trusting the rewrite. Every decision after
that point rests on the summary being accurate.

We audited 337 such summaries from 162 sessions. For each one the full transcript
it replaced was still available, which makes the comparison checkable rather than
impressionistic. Average preservation: **56%**.

## The finding

Loss is not uniform, and the ordering is the result.

| What the summary had to preserve | Preserved | Why |
|---|---:|---|
| Git commit hashes | 99.7% | A 7-character string. It matches or it does not |
| Order of events | 75.9% | Sequence, mostly mechanical |
| Which tools were used | 52.7% | Part list, part judgement of relevance |
| How many messages | 51.4% | A count, but of things worth mentioning |
| Which files changed | 47.8% | A list, filtered by what seemed important |
| What was actually said | 33.5% | Requires representing meaning, not tokens |
| What went wrong | 31.9% | Requires deciding what counted as going wrong |
| **Fabrication rate** | **0.0%** | Nothing invented, in any summary |

**Literal tokens survive compression. Interpretation does not.** A commit hash
needs no understanding to copy correctly. "What went wrong in this session"
requires deciding what counted as an error and how serious it was, and that is
what compression destroys.

The practical consequence: after a compaction the model is most reliable about
exactly the things you could have looked up yourself, and least reliable about
the things you would actually ask it.

## How much disappears

| | Happened | Survived | Lost |
|---|---:|---:|---:|
| Tool calls | 4,660 | 1,173 | 75% |
| User messages | 8,449 | 3,127 | 63% |
| Files changed | 2,400 | 1,106 | 54% |
| Errors hit | 7,672 | 4,576 | 40% |

Of 337 summaries, 263 under-report how many errors occurred and 57 over-report.
Combined with a zero fabrication rate, this is a model that drops bad news rather
than manufacturing it. A fact-checker aimed at invented claims would pass every
one of these summaries while the record quietly degraded.

**Most of the damage carries no warning label.** 207 of the 337 summaries, 61%,
trip none of the four named compaction pathologies, and their fidelity is no
better than the tagged ones. The tags catch specific recognisable distortions.
The broader loss is diffuse and silent.

## A note on the intervention figure

Partway through the window the working practice changed: sessions were ended
deliberately rather than left to run until the model was forced to compact.

Raw counts make this look decisive, peaking at 70 in March and falling to 1 in
April. Session volume over the same window ranges from 97 to 2,217, so raw counts
are not comparable month to month. Normalised per 100 sessions the drop is real
but June returns to roughly the December baseline, before the practice existed.

Figure 4 draws both panels deliberately, and the paper says the effect is not
established. Showing only the raw panel would have been the same overconfident
claim this work exists to catch.

## Rebuilding

```bash
pip install matplotlib python-pptx pandas duckdb
python make_figures.py     # reads figures/*.csv snapshots
python make_deck.py
pdflatex main && pdflatex main
```

`make_figures.py` prefers the live audit parquets when present and falls back to
the bundled CSV snapshots otherwise, which is what makes this directory
self-contained. Point `SESSION_DATALAKE` at a session-message store if you have
one; it is used only for the per-month denominators in figure 4.

## What this does not show

**One project, one system.** Research engineering sessions with one assistant.
The mechanism is general to anything that compresses its own context; the numbers
are local.

**The audit is automated.** Scores come from programmatic comparison of summary
claims against the transcript. Repeatable, and narrower than a human reading. A
summary can score well by naming the right things while still misleading about
their significance.

**Counts are a floor**, and the intervention analysis is correlational.

## License

GPL-3.0, per the repository root.
