#!/usr/bin/env python3
"""Five-slide compaction hallucination briefing.

Audience: people who have used generative AI and heard the term "hallucination"
but know little more. Target 5 to 10 minutes.

Numbers are read from the CSV snapshots written by make_figures.py, so the deck
cannot drift from the audit. Slide 6 is a marked backup carrying the
two-denominator analysis and can be deleted without affecting the talk.

Usage:
    python make_deck.py [--figures DIR] [--out FILE]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                          # noqa: E402

from deckkit import (AMBER, BLUE, BLUE_ON_DARK, Deck, GREEN,  # noqa: E402
                     GREEN_S, INK, MUTED, NAVY, RED, RED_S, SLATE, VIOLET,
                     VIOLET_S, WHITE, figure, footer, header, notes, rect,
                     table, text)

ROOT = Path(__file__).resolve().parents[2]


def load(figs):
    dims = pd.read_csv(figs / "fidelity_dimensions.csv")
    month = pd.read_csv(figs / "monthly_compaction.csv")
    agg = pd.read_csv(figs / "compaction_totals.csv").iloc[0].to_dict()
    return dims, month, agg


def build(figs, out):
    dims, month, agg = load(figs)
    n_sum, n_sess = int(agg["n_summaries"]), int(agg["n_sessions"])
    fid = agg["mean_fidelity"] * 100
    top = dims.iloc[0]
    bot = dims.iloc[-1]
    d = Deck()

    # ------------------------------------------------------------ 1 title ---
    s = d.slide()
    rect(s, 0, 0, 13.333, 2.6, NAVY)
    text(s, 0.9, 0.7, 11.5, 1.7,
         [("When the AI Rewrites Its Own History", 38, True, WHITE, 8),
          (f"Compaction hallucinations, measured across {n_sum} audited summaries",
           19, False, BLUE_ON_DARK, 0)])
    rect(s, 0.9, 3.3, 11.5, 1.6, RED_S, RED)
    text(s, 1.25, 3.48, 10.8, 1.3,
         [("A long session runs out of memory. The model summarises itself, "
           "throws away the original, and keeps going.", 18, True, INK, 6),
          (f"Those summaries preserved {fid:.0f}% of what actually happened.",
           21, True, RED, 0)])
    text(s, 0.9, 5.7, 11.5, 0.8,
         [("A 5-minute orientation, not a full treatment.", 15, False, MUTED, 4),
          ("Johnny Morgan   ·   University of Maryland, Baltimore County",
           13, False, MUTED, 0)])
    notes(s, f"""
The hook is that this is the model losing its OWN work, not losing a fact
somebody told it.

{n_sum} summaries from {n_sess} sessions. For each one we still had the full
transcript it replaced, so this is checkable rather than impressionistic.

{fid:.0f}% average preservation. Let that sit before moving on. Nearly half of
what happened did not survive into the record the model then worked from.
""")

    # ------------------------------------------------------ 2 what it is ----
    s = d.slide()
    header(s, "What compaction is", accent=AMBER, kicker="the mechanism")
    figure(s, figs / "fig1_what_is_compaction.png", 2.05, max_w=11.6, max_h=4.6)
    footer(s, 1, "The mechanism")
    notes(s, """
Walk the four boxes left to right. Long session, memory fills, it summarises
ITSELF, work continues from the summary.

The distinction that matters: this is not forgetting something you told it. It
is rewriting the record of its own work and then trusting the rewrite. Every
decision after that point rests on the summary being right.

The green box is the surprise, and it is worth pausing on. Zero fabrication
across every audited summary. It does not invent. It loses and it reshapes.
""")

    # -------------------------------------------------------- 3 gradient ----
    s = d.slide()
    header(s, "What survives, and what does not",
           "The ordering is the finding, not a presentation choice",
           accent=RED, kicker="the result")
    figure(s, figs / "fig2_fidelity_gradient.png", 2.05, max_w=12.1, max_h=4.55)
    footer(s, 2, "The result")
    notes(s, f"""
This is the slide. Give it time.

Top: git commit hashes at {top.score * 100:.1f}%. A seven-character string. Copying
it correctly requires no understanding, and it either matches or it does not.

Bottom: "{bot.dimension}" at {bot.score * 100:.0f}%. That requires deciding what
counted as going wrong, how serious it was, whether it got resolved. That is
interpretation.

The rule: the more a fact is a literal token, the better it survives. The more it
needs judgement, the worse.

The consequence to say out loud: after a compaction the model is most reliable
about the things you could have looked up yourself, and least reliable about the
things you would actually ask it.
""")

    # --------------------------------------------------------- 4 dropped ----
    s = d.slide()
    header(s, "How much simply disappears",
           f"Totals across all {n_sum} audited summaries",
           accent=BLUE, kicker="the loss")
    figure(s, figs / "fig3_what_gets_dropped.png", 2.0, max_w=11.6, max_h=3.9)
    rect(s, 0.9, 6.02, 11.5, 0.82, VIOLET_S, VIOLET)
    text(s, 1.25, 6.13, 10.9, 0.68,
         [("Most of the damage carries no warning label.", 14.5, True, INK, 2),
          (f"{int(agg['n_untagged'])} of {n_sum} summaries trip none of our four "
           "pathology detectors, and their fidelity is no better than the ones "
           "that do.", 13, False, INK, 0)])
    footer(s, 3, "The loss")
    notes(s, f"""
Three quarters of tool calls and nearly two thirds of user messages do not reach
the summary.

The direction matters: {int(agg['n_understate'])} summaries under-report how many
errors occurred, {int(agg['n_overstate'])} over-report. Combined with zero
fabrication, this is a model that drops bad news rather than manufacturing it.

That shapes the defence. A fact-checker aimed at invented claims would pass every
one of these summaries while the record quietly degraded.

The purple box is the uncomfortable part. We tag four named pathologies. 61% of
summaries trip none of them and are no more accurate. The loss is diffuse and
silent, not concentrated in flagged events.
""")

    # ------------------------------------------------------- 5 takeaways ----
    s = d.slide()
    rect(s, 0, 0, 13.333, 1.9, NAVY)
    text(s, 0.9, 0.55, 11.5, 1.1,
         [("What to do about it", 32, True, WHITE, 0)])
    for i, (n_, h, b, c) in enumerate([
        ("1", "Treat a post-compaction summary as a lead, not a record",
         "Reliable about identifiers, unreliable about interpretation. If a decision "
         "turns on what went wrong earlier, go to the source.", RED),
        ("2", "Anything that must survive should be a symbol",
         f"Commit hashes survived at {top.score * 100:.1f}% because they are literal "
         "strings with no interpretation in the middle. Exact paths, identifiers and "
         "error text survive compression. Prose does not.", GREEN),
        ("3", "End long sessions on purpose",
         "A session that stops before it is forced to compact never produces this "
         "failure at all. Doing so here coincided with a drop, though the effect is "
         "not yet established.", BLUE)]):
        top_y = 2.25 + i * 1.35
        rect(s, 0.9, top_y, 0.5, 0.62, c)
        text(s, 0.9, top_y + 0.06, 0.5, 0.5, [(n_, 17, True, WHITE, 0)], align=1)
        text(s, 1.6, top_y - 0.05, 10.9, 1.3,
             [(h, 17, True, INK, 3), (b, 13.5, False, MUTED, 0)])
    rect(s, 0.9, 6.08, 11.5, 0.76, GREEN_S, GREEN)
    text(s, 1.25, 6.18, 10.9, 0.62,
         [("Do not build the defence around fabrication. Zero invented content "
           "across all 337 summaries. The failure is omission, not invention.",
           13.5, True, INK, 0)])
    footer(s, 4, "Takeaways")
    notes(s, """
Land the three, then open the floor.

Point 3 needs its caveat said out loud, not buried. The raw counts look like a
collapse after the practice changed. Once you normalise by session volume, which
swings more than twentyfold across the window, the drop is real but partly
reverts. Backup slide has both panels if anyone pushes.

Open questions if there is appetite:
  - can a constrained summary format raise fidelity? testable against this audit
  - what happens across repeated compactions, a summary of a summary?
  - does the model know it lost something? nothing in these summaries flags it
""")

    # ---------------------------------------------------------- 6 backup ----
    s = d.slide()
    header(s, "Backup: did anything reduce it?",
           "Not part of the 5-slide talk. The same data under two denominators.",
           accent=MUTED, kicker="backup")
    figure(s, figs / "fig4_intervention.png", 2.05, max_w=12.1, max_h=4.4)
    footer(s, "B", "Backup")
    notes(s, """
Hold this for "so did the fix work?"

Left panel is what a deck usually shows: 70 events in March, 1 in April. Clean
collapse.

Right panel is what the left one omits. Session volume ranges from 97 to 2,217
across these months, so raw counts are not comparable. Normalised, March is still
the peak and April still falls, but June returns to roughly where December sat,
before the practice existed.

Supportable statement: it became less frequent, and the effect is not
established. A matched comparison is open work.

If it is useful, say why both panels are here: showing only the left one would be
the same overconfident claim this whole taxonomy exists to catch.
""")

    return d.save(out)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", default=str(here / "figures"))
    ap.add_argument("--out",
                    default=str(here / "Compaction_Hallucinations_Briefing.pptx"))
    args = ap.parse_args()
    figs = Path(args.figures)
    if not (figs / "fidelity_dimensions.csv").exists():
        raise SystemExit(f"missing snapshots in {figs}. Run make_figures.py first.")
    n, path = build(figs, Path(args.out))
    print(f"{n} slides (5 + 1 backup) -> {path}  "
          f"({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
