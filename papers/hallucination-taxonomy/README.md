# A Taxonomy of AI Hallucinations

Eight classes and fifty-one types, found by measuring real work.

**Johnny Morgan**, University of Maryland, Baltimore County

| File | What |
|---|---|
| `main.pdf` | The paper, 9 pages, includes all 51 types with definitions |
| `main.tex` | LaTeX source |
| `Hallucination_Taxonomy_Briefing.pptx` | 5 slides plus one marked backup, speaker notes throughout |
| `figures/` | 4 figures at 200 dpi |
| `figures/taxonomy_snapshot.csv` | Every number on every slide, one row per type |
| `make_figures.py` | Regenerates the figures from the snapshot |
| `make_deck.py` | Regenerates the deck from those figures |
| [`../figkit.py`](../figkit.py), [`../deckkit.py`](../deckkit.py) | Shared palette and layout primitives |

## The short version

Most people who have used a generative AI system have heard the word
"hallucination" and take it to mean the model made something up. That is part of
it. It is not most of it.

The working definition this taxonomy is built on:

> A hallucination is output the model presents with confidence that is not
> grounded in anything real: not in the data it was given, not in the tools it
> has, and not in what actually happened.

Measured across an extended body of real work, 1,690 detected and confirmed
failure events fall into 8 classes and 51 types.

| Class | In one line | Types | Events |
|---|---|---:|---:|
| Fabrication | States something that is not so | 11 | 492 |
| Wrong method | Gets there the wrong way | 6 | 433 |
| Reinvention | Rebuilds what already exists | 4 | 274 |
| Behavioural | Skips the steps it was told to take | 10 | 268 |
| Memory loss | Forgets, or misremembers, its own session | 8 | 170 |
| Architecture | Ignores how the system is built | 3 | 30 |
| Stale data | Trusts information that has expired | 3 | 17 |
| Pipeline integrity | Breaks the data on the way through | 6 | 6 |
| **Total** | | **51** | **1,690** |

**Fabrication is 29% of events.** It is the largest class and still under a
third. A programme that addresses only fabrication addresses under a third of
the problem while reporting that hallucination is solved.

**The two most frequent individual types invent nothing.** The most common,
394 events, is the model using a slow method when a fast one was available.
Nothing it produced was false; the work was simply done the wrong way. The
second, 230 events, is rebuilding a tool that already existed.

## How these were found

The taxonomy was not designed and then looked for. It accumulated the other way
round. An AI assistant worked a long-running research project across hundreds of
sessions with everything recorded. When something went wrong it was named,
defined, and added to a registry, and detection was automated where possible so
the same failure could be counted rather than merely remembered.

## What it does not show

**One project, one kind of work.** These are research engineering sessions.
A customer service assistant or a summariser would show a different mix. Treat
the classes as portable and the percentages as local.

**Detected, not total.** Every count is a floor. The types with the highest
confirmation counts are the ones reality can contradict most directly, so the
taxonomy counts checkable failures better than subtle ones. A low number can
mean a rare failure or a hard-to-see one, and the table does not distinguish
them.

**Counts depend on which record you use.** 1,690 is the observed-event count in
the detection matrix. Other records in the same system answer slightly different
questions and give different totals. Any number quoted from this work should
name the record it came from.

## Rebuilding

```bash
pip install matplotlib python-pptx pandas duckdb
python make_figures.py     # reads figures/taxonomy_snapshot.csv
python make_deck.py        # builds the .pptx from those figures
pdflatex main && pdflatex main
```

`make_figures.py` prefers a live inventory parquet when one is present and falls
back to the bundled CSV snapshot otherwise, which is what makes this directory
self-contained.

## A note on wording

The source definitions were written in situ and named the specific assistant
that produced each failure. Every type here is a property of a generative model
under agentic use rather than of one vendor's product, so the published
definitions say "the model". This is a presentation change only; no count,
category, or type name was altered.

## License

GPL-3.0, per the repository root.
