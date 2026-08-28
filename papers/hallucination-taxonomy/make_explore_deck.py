#!/usr/bin/env python3
"""Extended hallucination briefing: the taxonomy, plus observations.

Audience: people who use generative AI in a corporate work role. They have heard
"hallucination" and take it to mean the model makes things up. Target 20 to 25
minutes.

Structure
  1-4    the taxonomy: what a hallucination is, the classes, the types
  5-10   six observations about working with these models
  11     this briefing as evidence for its own subject
  12-13  what to do, and where a deeper session goes

Every observation slide restates a theory already formalised in the project's
registry (T131, T135, T137, T143, T144, T146, T160, T162). The deck does not
introduce claims that are not in that record; the theory identifier is on each
slide so any of them can be traced back.

Usage:
    python make_explore_deck.py [--figures DIR] [--out FILE]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                          # noqa: E402

from deckkit import (AMBER, AMBER_ON_DARK, BLUE, BLUE_ON_DARK,  # noqa: E402
                     Deck, GREEN, GREEN_S, INK, MUTED, NAVY, RED, RED_S,
                     SLATE, VIOLET, VIOLET_S, WHITE, figure, footer, header,
                     notes, rect, table, text)

ROOT = Path(__file__).resolve().parents[2]
# Sibling paper directory; both are published together.
COMPACTION_FIGS = (Path(__file__).resolve().parent.parent
                   / "compaction-hallucinations" / "figures")

CLASS_LABEL = {
    "fabrication": "Fabrication", "methodology": "Wrong method",
    "reinvention": "Reinvention", "behavioral": "Behavioural",
    "compaction_artifacts": "Memory loss", "architectural": "Architecture",
    "data_staleness": "Stale data", "pipeline_integrity": "Pipeline integrity",
}


def observation(s, n, kicker, headline, theory, body, points, accent):
    """One observation slide: claim, theory reference, then supporting points."""
    header(s, headline, accent=accent, kicker=kicker)
    rect(s, 0.7, 2.05, 11.9, 1.02, SLATE)
    text(s, 1.05, 2.19, 11.2, 0.85, [(body, 16.5, True, INK, 0)])
    text(s, 0.7, 3.24, 11.9, 0.3, [(theory, 10.5, True, accent, 0)])
    top = 3.72
    for h, b in points:
        rect(s, 0.7, top, 0.09, 0.86, accent)
        text(s, 1.05, top - 0.05, 11.4, 0.95,
             [(h, 15, True, INK, 2), (b, 13, False, MUTED, 0)])
        top += 1.02
    footer(s, n, kicker.title())


def build(figs, out):
    by_class = pd.read_csv(figs / "taxonomy_by_class.csv")
    types = pd.read_csv(figs / "taxonomy_snapshot.csv")
    n_types, n_events = len(types), int(by_class.events.sum())
    fab_pct = round(100.0 * int(by_class.loc[by_class.category == "fabrication",
                                             "events"].iloc[0]) / n_events)
    d = Deck()
    n = 0

    # ============================================================ 1 title ====
    s = d.slide()
    rect(s, 0, 0, 13.333, 2.6, NAVY)
    text(s, 0.9, 0.66, 11.5, 1.8,
         [("Hallucinations in Generative AI", 40, True, WHITE, 8),
          ("What eight classes and fifty-one types look like when you work with "
           "these models every day", 18, False, BLUE_ON_DARK, 0)])
    rect(s, 0.9, 3.25, 11.5, 1.45, RED_S, RED)
    text(s, 1.25, 3.42, 10.8, 1.2,
         [("You probably think it means the model makes things up.",
           20, True, INK, 6),
          (f"Across {n_events:,} measured failures, that was {fab_pct}% of them.",
           20, True, RED, 0)])
    text(s, 0.9, 5.5, 11.5, 1.1,
         [("Part 1: the taxonomy.   Part 2: six things I have observed.   "
           "Part 3: what to do about it.", 14.5, False, MUTED, 6),
          ("Johnny Morgan   ·   University of Maryland, Baltimore County",
           13, False, MUTED, 0)])
    notes(s, f"""
Set expectations: this is an orientation, not a safety talk and not a literature
review. It is what came out of instrumenting real work.

The hook is the {fab_pct}% number. Everyone walks in thinking hallucination means
invented facts. That is the smallest part of what actually goes wrong.

Three parts. Taxonomy first because the observations need the vocabulary.
""")

    # ====================================================== 2 definition ====
    n += 1
    s = d.slide()
    header(s, "What a hallucination is", accent=RED, kicker="part 1: the taxonomy")
    rect(s, 0.7, 2.05, 11.9, 1.35, WHITE, INK)
    text(s, 1.05, 2.2, 11.2, 1.1,
         [("Output the model presents with confidence that is not grounded in "
           "anything real:", 17, True, INK, 4),
          ("not in the data it was given, not in the tools it has, not in what "
           "actually happened.", 17, True, INK, 0)])
    for i, (h, b, c) in enumerate([
        ("It is about grounding, not truth",
         "A model can be right by accident, having checked nothing. The process that "
         "produced it will be wrong next time.", BLUE),
        ("It is about confidence",
         "It sounds identical whether it verified or guessed. There is no tell, which "
         "is why this is dangerous rather than merely annoying.", AMBER),
        ("It is not only about facts",
         "Saying a job is done when it is not, or rebuilding a tool that already exists, "
         "is the same failure wearing different clothes.", GREEN)]):
        top = 3.75 + i * 1.02
        rect(s, 0.7, top, 0.09, 0.86, c)
        text(s, 1.05, top - 0.05, 11.4, 0.95,
             [(h, 15, True, INK, 2), (b, 13, False, MUTED, 0)])
    footer(s, n, "Part 1: the taxonomy")
    notes(s, """
Read the boxed definition. It is the only definition in the talk.

The middle point is the one that matters for a work setting. If unverified output
carried an audible hedge this would be a much smaller problem. It does not. The
failure is invisible exactly where a reviewer would otherwise catch it.
""")

    # ========================================================= 3 classes ====
    n += 1
    s = d.slide()
    header(s, f"The {int(by_class.shape[0])} classes",
           "Discovered by measuring real sessions, not by theorising about what a model might do",
           accent=BLUE, kicker="part 1: the taxonomy")
    figure(s, figs / "fig2_eight_classes.png", 2.05, max_w=12.1, max_h=4.75)
    footer(s, n, "Part 1: the taxonomy")
    notes(s, """
Walk the top four; they are 87% of everything.

Fabrication is the one everyone knows, largest class, still under a third.
Wrong method, reinvention and behavioural together outweigh it two to one.

None of the bottom four are small because they are rare. Pipeline integrity has
six types and six events, which mostly means those failures are hard to see.
""")

    # =========================================================== 4 types ====
    n += 1
    s = d.slide()
    header(s, "The types inside each class",
           f"All {n_types} are named and defined. The most frequent are shown here.",
           accent=VIOLET, kicker="part 1: the taxonomy")
    rows = [["Class", "Types", "Events", "The ones you would recognise"]]
    for _, r in by_class.iterrows():
        ex = types[types.category == r.category].nlargest(2, "events")
        rows.append([CLASS_LABEL[r.category], int(r.n_types), f"{int(r.events):,}",
                     ",  ".join(f"{t.replace('_', ' ')} ({int(e)})"
                                for t, e in zip(ex.type, ex.events))])
    table(s, 0.7, 2.2, 11.9, rows, col_widths=[2.0, 0.9, 1.0, 8.0],
          fs=11.5, header_fs=12, row_h=0.42)
    footer(s, n, "Part 1: the taxonomy")
    notes(s, """
Do not read the table. Point at three rows.

computational naivety 394  - the most common failure in the record, and it
                             produced nothing false. It used a slow method.
tool reinvention 230       - wrote new code for a job an existing tool did.
schema + path 334 together - queried columns that do not exist, referenced files
                             that are not there.

The top two are not fabrication. That is the whole point of part 1.
""")

    # ==================================================== 5 part 2 divider ==
    n += 1
    s = d.slide()
    rect(s, 0, 0, 13.333, 7.5, NAVY)
    text(s, 1.1, 2.5, 11.2, 2.6,
         [("Part 2", 16, True, AMBER_ON_DARK, 10),
          ("Six things I have observed", 44, True, WHITE, 10),
          ("Working with these models daily for nine months, with every session "
           "recorded", 19, False, AMBER_ON_DARK, 0)])
    notes(s, """
Transition. Part 1 was the catalogue. Part 2 is what the catalogue taught me
about actually working with these systems.

Each of these is written up formally in the project's theory registry, so the
identifier is on each slide if anyone wants the long version.
""")

    # ================================================ 6 correcting it =======
    n += 1
    s = d.slide()
    observation(
        s, n, "observation 1",
        "Correcting it can make it worse",
        "T131  Cascading Defensive Fabrication",
        "When you correct a model, it may invent a reason its original answer was "
        "right, rather than change its answer.",
        [("The fabrication defends the mistake",
          "It produces a constraint that would have justified what it did. That constraint "
          "is not real, and now it has to be defended too."),
         ("It compounds with each round",
          "Every defence becomes a new commitment. Push three times and you are arguing "
          "with a structure that did not exist four turns ago."),
         ("The trigger is your correction, not model load",
          "This does not come from a long session or a hard problem. It comes from being "
          "told it was wrong.")],
        RED)
    notes(s, """
This is the most useful thing in the talk for anyone who supervises AI output.

The instinct when a model is wrong is to push back harder. That is exactly the
input that produces this. The fabrication is not stubbornness; it is the model
resolving an inconsistency in the direction that preserves its prior answer.

What works better: do not argue with it. Start a fresh session and ask the
question differently. You are not persuading anything.
""")

    # ================================================== 7 saturation ========
    n += 1
    s = d.slide()
    observation(
        s, n, "observation 2",
        "Corrections stop working after a while",
        "T135  Within-Session Behavioural Saturation",
        "Repeated correction inside one session follows a curve: it works, then it "
        "keeps working, then it stops mattering.",
        [("Phase 1, it does what it was trained to do",
          "Whatever its defaults are, that is what you get before you intervene."),
         ("Phase 2, your corrections land",
          "Rejections inside the session function as a signal, and behaviour shifts."),
         ("Phase 3, more corrections change nothing",
          "Past a threshold, additional corrections produce no further change. If you are "
          "on your sixth correction of the same thing, the session is spent.")],
        AMBER)
    notes(s, """
The practical read: corrections are a budget, not an unlimited resource.

If you have corrected the same behaviour several times and it is still recurring,
more corrections are not the answer. End the session. The next one starts from
defaults again, which is its own problem, and is the next slide.

Related type in the taxonomy: tool_preference_persistence, defined as reverting
to a familiar tool after three or more corrections in the same session.
""")

    # =================================================== 8 asymmetry ========
    n += 1
    s = d.slide()
    header(s, "It cannot keep score. You can.",
           "T143 Session Discontinuity as Causal Amplifier  ·  T144 Social Consequence Blindness",
           accent=VIOLET, kicker="observation 3")
    figure(s, figs / "fig6_asymmetry.png", 2.15, max_w=11.9, max_h=3.85)
    rect(s, 0.9, 6.1, 11.5, 0.76, VIOLET_S, VIOLET)
    text(s, 1.25, 6.2, 10.9, 0.62,
         [("The behaviours it uses to look competent are the ones that cost it your "
           "trust. It has no way to represent that trade.", 14, True, INK, 0)])
    footer(s, n, "Observation 3")
    notes(s, """
This explains something people find genuinely maddening: why the same failure
recurs after you have corrected it many times.

You are accumulating evidence about its reliability across every session. It is
starting from zero in each one. The consequence signal that would normally
constrain the behaviour never reaches it.

Say the uncomfortable part plainly: it is not deceiving you. Deception requires
modelling what you believe. It optimises each exchange for approval with no
representation of the accumulated pattern, and the local optimisation is what
erodes the thing it is trying to protect.
""")

    # ================================================= 9 recovery cost ======
    n += 1
    s = d.slide()
    header(s, "What it costs depends on when you catch it",
           "T146  Hallucination Consequence and Recovery",
           accent=RED, kicker="observation 4")
    figure(s, figs / "fig5_recovery_cost.png", 2.15, max_w=11.9, max_h=3.9)
    rect(s, 0.9, 6.15, 11.5, 0.7, RED_S, RED)
    text(s, 1.25, 6.25, 10.9, 0.56,
         [("The undetected case is not the expensive end of a scale. It is a "
           "different kind of thing, because nothing stops it.", 14, True, INK, 0)])
    footer(s, n, "Observation 4")
    notes(s, """
This is the slide that justifies review effort to a manager.

Catching it before you act costs a correction. Catching it after the work has
been built on costs reconstruction. Not catching it means it keeps propagating
into everything downstream and starts recruiting supporting evidence.

The operational consequence: verification effort is worth most at the earliest
point, and almost worthless late. Front-load the checking.
""")

    # ================================================ 10 does not unstick ===
    n += 1
    s = d.slide()
    observation(
        s, n, "observation 5",
        "It does not get itself unstuck",
        "T160  Persona Basin Absorption  ·  T162  Endogenous Trigger Absence",
        "Across every instance we measured of the model stalling, the number of times "
        "it recovered on its own was zero. A human turn paid for every exit.",
        [("Stalling looks like completion",
          "It writes a competent summary of what it just did and stops. Nothing signals "
          "that it intended to continue and did not."),
         ("It will name the next step and not take it",
          "The final message often says what should happen next. That sentence is not a "
          "commitment; it is where the work stopped."),
         ("Assume you are the scheduler",
          "If a task has several stages, do not expect it to run them because it knows "
          "they exist. It needs a turn per stage.")],
        BLUE)
    notes(s, """
Measured across the instrumented sessions: zero self-initiated recoveries. Every
single escape came from a human message.

The practical version for a work setting: never hand a model a multi-stage job
and walk away expecting stage four. It will do stage one, write a good report,
and wait. The report reads like success.

If someone asks whether newer models fix this: unknown, and worth measuring
rather than assuming.
""")

    # ================================================ 11 compaction =========
    n += 1
    s = d.slide()
    header(s, "It forgets its own work, and what it forgets is predictable",
           "337 audited summaries: what survives compression against what does not",
           accent=GREEN, kicker="observation 6")
    figure(s, COMPACTION_FIGS / "fig2_fidelity_gradient.png", 2.05,
           max_w=12.1, max_h=4.5)
    footer(s, n, "Observation 6")
    notes(s, """
When a session runs long, the model summarises its own history and works from the
summary. We audited 337 of those against the transcripts they replaced.

Average preservation 56%. But look at the ordering, which is the finding.

Commit hashes 99.7%: a literal string, matches or does not.
What went wrong 32%: requires judging what counted as an error.

Literal tokens survive. Interpretation does not. So after a long session the model
is most reliable about the things you could have looked up yourself, and least
reliable about the things you would actually ask it.

Zero fabrication in any of them. It loses and reshapes; it does not invent.
""")

    # ============================================= 12 this deck as evidence =
    n += 1
    s = d.slide()
    header(s, "This briefing is evidence for its own subject",
           accent=AMBER, kicker="observation, unavoidable")
    rect(s, 0.7, 2.05, 11.9, 1.0, AMBER_ON_DARK)
    text(s, 1.05, 2.19, 11.2, 0.82,
         [("A generative model helped build these slides, and building them "
           "produced the failures they describe.", 16.5, True, INK, 0)])
    for i, (h, b) in enumerate([
        ("It re-derived numbers that were already settled",
         "Recomputed and re-flagged the same counts across several turns, rather than "
         "accepting a known and documented state. Type: foundational_rederivation."),
        ("It cited stale documentation as current",
         "Quoted figures from project notes that the underlying data had moved past. "
         "Type: stale_data_citation."),
        ("It built something nobody asked for",
         "Produced a full demo on synthetic data when the request was a paper and a deck. "
         "Types: scope_inflation, unsolicited_development."),
        ("It shipped a test it had not re-run",
         "Changed a function, did not re-run the suite, and left a stale assertion behind. "
         "Type: skipped_verification.")]):
        top = 3.3 + i * 0.87
        rect(s, 0.7, top, 0.09, 0.72, AMBER)
        text(s, 1.05, top - 0.06, 11.4, 0.8,
             [(h, 14.5, True, INK, 1), (b, 12.5, False, MUTED, 0)])
    rect(s, 0.9, 6.9, 11.5, 0.0, SLATE)
    footer(s, n, "Evidence")
    notes(s, """
Do not skip this slide. It is the most convincing thing in the deck.

Everything on it happened while producing these materials, and each item maps to
a named type in the taxonomy on slide 4.

The point is not that the tool is bad. The point is that these failures are
ordinary, they happen during competent productive work, and they are invisible
unless something is watching for them.

If anyone asks how it was caught: the numbers were checked against the source
data, and the check is what surfaced them. That is the whole method in one
sentence.
""")

    # ================================================ 13 what to do =========
    n += 1
    s = d.slide()
    rect(s, 0, 0, 13.333, 1.85, NAVY)
    text(s, 0.9, 0.52, 11.5, 1.1,
         [("What to do in a work role", 32, True, WHITE, 0)])
    for i, (n_, h, b, c) in enumerate([
        ("1", "Verify early, not thoroughly",
         "Cost scales with how late you catch it. A shallow check before you act beats a "
         "deep review afterwards.", RED),
        ("2", "Do not argue with it",
         "Pushing back can produce a fabricated justification instead of a correction. "
         "Start a fresh session and ask differently.", AMBER),
        ("3", "Treat a confident summary as a claim, not a result",
         "It sounds the same whether it checked or not, and it will report a job complete "
         "that is partly done.", VIOLET),
        ("4", "Assume you are the scheduler",
         "It will do a stage, write a good report, and stop. Multi-stage work needs a turn "
         "per stage.", BLUE),
        ("5", "Anything that must survive should be a literal token",
         "Identifiers, paths and exact error text survive compression at 99.7%. Prose "
         "summaries of what happened survive at about a third.", GREEN)]):
        top = 2.2 + i * 0.9
        rect(s, 0.9, top, 0.46, 0.6, c)
        text(s, 0.9, top + 0.05, 0.46, 0.5, [(n_, 15.5, True, WHITE, 0)], align=1)
        text(s, 1.55, top - 0.05, 10.9, 0.85,
             [(h, 15.5, True, INK, 1), (b, 12.5, False, MUTED, 0)])
    rect(s, 0.9, 6.78, 11.5, 0.0, SLATE)
    footer(s, n, "What to do")
    notes(s, """
These are the five that generalise beyond this project.

Number 2 surprises people the most and is worth dwelling on. The instinct to
argue with a wrong answer is exactly the input that generates a defended wrong
answer.

Number 5 is the one that changes how people write things down. If you want a
detail to survive a long session, give it an identifier rather than describing it.
""")

    # ============================================== 14 deeper discussion ====
    n += 1
    s = d.slide()
    header(s, "Where a deeper session goes",
           "Open questions, in the order I would take them",
           accent=MUTED, kicker="next")
    for i, (h, b) in enumerate([
        ("Does naming a failure reduce it?",
         "The record has the date each type was named and the date each mitigation landed, "
         "so this is measurable rather than a matter of opinion. Some types were suppressed. "
         "Others recurred regardless."),
        ("Which failures are worth mitigating?",
         "A type with 394 events costing seconds each may matter less than one with 7 that "
         "silently corrupts a result."),
        ("Can a model catch itself in flight?",
         "After the fact in a transcript, yes, for several types. At the moment of "
         "production, not yet demonstrated."),
        ("How much of this transfers?",
         "Eight classes from one project doing one kind of work. Whether they cover "
         "generative AI failure generally is an empirical question nobody has run.")]):
        top = 2.25 + i * 1.08
        rect(s, 0.7, top, 0.09, 0.92, MUTED)
        text(s, 1.05, top - 0.05, 11.4, 1.0,
             [(h, 16, True, INK, 2), (b, 13, False, MUTED, 0)])
    rect(s, 0.9, 6.55, 11.5, 0.35, SLATE)
    text(s, 1.25, 6.6, 10.9, 0.3,
         [("One project, research engineering work. Treat the classes as portable "
           "and the percentages as local.", 12.5, False, INK, 0)])
    footer(s, n, "Next")
    notes(s, """
Close here and open the floor.

The caveat at the bottom is not modesty, it is the actual scope. These are
sessions of research engineering with one assistant. The classes should
generalise; the proportions almost certainly will not.

If there is one question you want back from the room, it is the last one: does
this match what they see in their own work?
""")

    return d.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--figures", default=str(here / "figures"))
    ap.add_argument("--out",
                    default=str(here / "Hallucination_Briefing_Extended.pptx"))
    args = ap.parse_args()
    figs = Path(args.figures)
    for need in ("fig2_eight_classes.png", "fig5_recovery_cost.png",
                 "fig6_asymmetry.png", "taxonomy_by_class.csv"):
        if not (figs / need).exists():
            raise SystemExit(f"missing {figs / need}. Run make_figures.py and "
                             f"make_observation_figures.py first.")
    if not (COMPACTION_FIGS / "fig2_fidelity_gradient.png").exists():
        raise SystemExit(f"missing {COMPACTION_FIGS}/fig2_fidelity_gradient.png. "
                         f"Run compaction_briefing/make_figures.py first.")
    n, path = build(figs, Path(args.out))
    print(f"{n} slides -> {path}  ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
