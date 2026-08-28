#!/usr/bin/env python3
"""Five-slide hallucination taxonomy briefing.

Audience: people who have used generative AI and have heard the word
"hallucination" but know little more. Target 5 to 10 minutes.

Numbers come from taxonomy_by_class.csv, written by make_figures.py from the
live inventory parquet, so the deck cannot drift from the data. Slide 6 is a
clearly-marked backup and can be deleted without affecting the talk.

Usage:
    python make_deck.py [--figures DIR] [--out FILE]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                          # noqa: E402

from deckkit import (AMBER, BLUE, BLUE_ON_DARK, Deck, GREEN,  # noqa: E402
                     GREEN_ON_DARK, GREEN_S, INK, MUTED, NAVY, RED, RED_S,
                     SLATE, VIOLET, VIOLET_S, WHITE, figure, footer, header,
                     notes, rect, table, text)

ROOT = Path(__file__).resolve().parents[2]

CLASS_LABEL = {
    "fabrication": "Fabrication", "methodology": "Wrong method",
    "reinvention": "Reinvention", "behavioral": "Behavioural",
    "compaction_artifacts": "Memory loss", "architectural": "Architecture",
    "data_staleness": "Stale data", "pipeline_integrity": "Pipeline integrity",
}
CLASS_GLOSS = {
    "fabrication": "States something that is not so",
    "methodology": "Gets there the wrong way",
    "reinvention": "Rebuilds what already exists",
    "behavioral": "Skips the steps it was told to take",
    "compaction_artifacts": "Forgets its own session",
    "architectural": "Ignores how the system is built",
    "data_staleness": "Trusts information that has expired",
    "pipeline_integrity": "Breaks the data on the way through",
}


def load(figs):
    by_class = pd.read_csv(figs / "taxonomy_by_class.csv")
    types = pd.read_csv(figs / "taxonomy_snapshot.csv")
    tot = {
        "types": int(len(types)),
        "classes": int(by_class.shape[0]),
        "events": int(by_class.events.sum()),
    }
    fab = int(by_class.loc[by_class.category == "fabrication", "events"].iloc[0])
    tot["fab_pct"] = round(100.0 * fab / tot["events"])
    return by_class, types, tot


def examples(types, cat, n=3):
    """Highest-volume types in a class, as 'name (count)' for the slide table."""
    rows = types[types.category == cat].nlargest(n, "events")
    return ",  ".join(f"{t.replace('_', ' ')} ({int(e)})"
                      for t, e in zip(rows.type, rows.events))


def build(figs, out):
    by_class, types, tot = load(figs)
    d = Deck()

    # ------------------------------------------------------------ 1 title ---
    s = d.slide()
    rect(s, 0, 0, 13.333, 2.6, NAVY)
    text(s, 0.9, 0.7, 11.5, 1.7,
         [("What AI Hallucinations Actually Are", 40, True, WHITE, 8),
          (f"{tot['classes']} classes, {tot['types']} types, "
           f"{tot['events']:,} measured events", 19, False, BLUE_ON_DARK, 0)])
    rect(s, 0.9, 3.3, 11.5, 1.5, RED_S, RED)
    text(s, 1.25, 3.5, 10.8, 1.2,
         [("You probably think it means the AI makes things up.", 21, True, INK, 6),
          (f"In {tot['events']:,} measured failures, that was {tot['fab_pct']}% of them.",
           21, True, RED, 0)])
    text(s, 0.9, 5.6, 11.5, 0.8,
         [("A 5-minute orientation, not a full treatment.", 15, False, MUTED, 4),
          ("Johnny Morgan   ·   University of Maryland, Baltimore County",
           13, False, MUTED, 0)])
    notes(s, """
Open with the hook and let it sit for a beat.

Everyone in the room has heard "hallucination" and thinks it means the model
invented a fact. That is real, and it is under a third of what we measured. The
other two thirds is the model doing the work wrong while sounding exactly as
confident as when it does the work right.

Say what this is NOT: not a survey of the literature, not a safety talk. It is
what we found by instrumenting our own work.
""")

    # -------------------------------------------------------- 2 definition ---
    s = d.slide()
    header(s, "What a hallucination is", accent=RED, kicker="the definition")
    rect(s, 0.7, 2.05, 11.9, 1.5, WHITE, INK)
    text(s, 1.05, 2.2, 11.3, 1.25,
         [("Output the model presents with confidence that is not grounded in "
           "anything real:", 18, True, INK, 4),
          ("not in the data it was given, not in the tools it has, not in what "
           "actually happened.", 18, True, INK, 0)])

    for i, (h, b, c) in enumerate([
        ("It is about grounding, not truth",
         "A model can be right by accident, having checked nothing. The process "
         "that produced it will be wrong next time.", BLUE),
        ("It is about confidence",
         "It sounds the same whether it verified or guessed. That is why the "
         "failure is invisible where you would catch it.", AMBER),
        ("It is not only about facts",
         "Saying a job is done when it is not, or rebuilding a tool that exists, "
         "is the same failure wearing different clothes.", GREEN)]):
        top = 3.95 + i * 0.98
        rect(s, 0.7, top, 0.09, 0.82, c)
        text(s, 1.05, top - 0.04, 11.4, 0.9,
             [(h, 16, True, INK, 2), (b, 13.5, False, MUTED, 0)])
    footer(s, 1, "The definition")
    notes(s, """
Read the boxed definition out loud. It is the only definition in the talk.

The three points underneath are the ones people push back on:

Grounding not truth   - "but it was right!" It was right by luck. Same process,
                        different day, wrong answer.
Confidence            - this is why it is dangerous rather than merely annoying.
                        There is no tell.
Not only facts        - this is the bridge to the next slide, where 7 of the 8
                        classes are not about inventing facts at all.
""")

    # ----------------------------------------------------------- 3 classes ---
    s = d.slide()
    header(s, f"The {tot['classes']} classes we found",
           "Discovered by measuring real working sessions, not by theorising",
           accent=BLUE, kicker="what we found")
    figure(s, figs / "fig2_eight_classes.png", 2.05, max_w=12.1, max_h=4.75)
    footer(s, 2, "The classes")
    notes(s, """
Walk the top four, they are 87% of everything.

Fabrication 492   - the one everyone knows. Largest class, still under a third.
Wrong method 433  - the model gets the right answer the slow way.
Reinvention 274   - it rebuilds something that already existed.
Behavioural 268   - it skips the steps it was told to take.

The bottom four are small in count. Pipeline integrity has 6 types and 6 events,
which mostly means those failures are hard to see, not that they are rare.

If someone asks where the classes came from: they were not designed up front.
Each was named the first time it happened, then detection was automated so it
could be counted rather than remembered.
""")

    # ------------------------------------------------------------ 4 table ---
    s = d.slide()
    header(s, "The types inside each class",
           f"All {tot['types']} are named and defined. The most frequent are shown here.",
           accent=VIOLET, kicker="the breakout")
    rows = [["Class", "Types", "Events", "The ones you would recognise"]]
    for _, r in by_class.iterrows():
        rows.append([CLASS_LABEL[r.category], int(r.n_types), f"{int(r.events):,}",
                     examples(types, r.category, 2)])
    table(s, 0.7, 2.2, 11.9, rows, col_widths=[2.0, 0.9, 1.0, 8.0],
          fs=11.5, header_fs=12, row_h=0.42)
    footer(s, 3, "The breakout")
    notes(s, """
Do not read the table. Point at three rows.

computational naivety, 394   - the single most common failure in the whole
                               record, and nothing it produced was false. It
                               used a slow method when a fast one existed.
tool reinvention, 230        - second most common. It wrote new code to do a job
                               an existing tool already did.
schema + path hallucination  - 334 between them. Querying columns that do not
                               exist, referencing files that are not there.
                               These are the most confirmable failures we have,
                               because a path either exists or it does not.

The point to land: the top two are not fabrication.
""")

    # -------------------------------------------------------- 5 takeaways ---
    s = d.slide()
    rect(s, 0, 0, 13.333, 1.9, NAVY)
    text(s, 0.9, 0.55, 11.5, 1.1,
         [("Three things to take away", 32, True, WHITE, 0)])
    for i, (n_, h, b, c) in enumerate([
        ("1", "It is not mostly making things up",
         f"Fabrication is {tot['fab_pct']}% of measured events. A programme that "
         "addresses only fabrication addresses under a third of the problem while "
         "reporting that hallucination is solved.", RED),
        ("2", "The most common failure costs time, not truth",
         "The single largest type produced nothing false. It did the work the slow "
         "way. That will not show up in any accuracy metric.", VIOLET),
        ("3", "Naming a failure is the cheap part",
         f"All {tot['types']} types are named. 30 have a mitigation aimed at them. "
         "Detecting, reducing, and proving the reduction get harder in that order.", GREEN)]):
        top = 2.25 + i * 1.35
        rect(s, 0.9, top, 0.5, 0.62, c)
        text(s, 0.9, top + 0.06, 0.5, 0.5, [(n_, 17, True, WHITE, 0)],
             align=1)
        text(s, 1.6, top - 0.05, 10.9, 1.3,
             [(h, 17, True, INK, 3), (b, 13.5, False, MUTED, 0)])
    rect(s, 0.9, 6.08, 11.5, 0.76, GREEN_S, GREEN)
    text(s, 1.25, 6.18, 10.9, 0.62,
         [("Where a deeper session goes: does naming reduce it, which ones are "
           "worth fixing, and can a model catch itself in flight?",
           13.5, True, INK, 0)])
    footer(s, 4, "Takeaways")
    notes(s, """
Land on the three, then open the floor.

If there is time and appetite, the four open questions are:
  1. Does naming a failure reduce it? We can measure this: the record has the
     date each type was named and the date each mitigation landed.
  2. Which are worth mitigating? 394 cheap events may matter less than 7 that
     silently corrupt a result.
  3. Can a model detect its own hallucinations in flight? After the fact in a
     transcript, yes, for several types. At the moment of production, no.
  4. How much of this transfers beyond one project? Nobody has run that.

Caveat to offer if challenged: this is one project's sessions, research
engineering work. Treat the classes as portable and the percentages as local.
Every count is a floor, since it counts what we detected.
""")

    # ----------------------------------------------------------- 6 backup ---
    s = d.slide()
    header(s, "Backup: the most frequent types",
           "Not part of the 5-slide talk. Delete or keep for questions.",
           accent=MUTED, kicker="backup")
    figure(s, figs / "fig3_type_breakout.png", 2.05, max_w=12.0, max_h=4.7)
    footer(s, "B", "Backup")
    notes(s, """
Hold this for the question "so what does it actually look like?"

Colour is the class. The two longest bars are purple and orange, wrong method
and reinvention. Red, fabrication, does not appear until third place.
""")

    return d.save(out)


def main():
    here = Path(__file__).resolve().parent
    default_figs = here / "figures"
    default_out = here / "Hallucination_Taxonomy_Briefing.pptx"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figures", default=str(default_figs))
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args()

    figs = Path(args.figures)
    if not (figs / "taxonomy_by_class.csv").exists():
        raise SystemExit(f"missing {figs}/taxonomy_by_class.csv. "
                         f"Run make_figures.py first.")
    n, path = build(figs, Path(args.out))
    print(f"{n} slides (5 + 1 backup) -> {path}  "
          f"({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
