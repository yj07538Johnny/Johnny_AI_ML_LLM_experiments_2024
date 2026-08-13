#!/usr/bin/env python3
"""build_deck.py - render the whitepaper figures to PNG and assemble a LEADERSHIP
briefing PowerPoint: a tight main line (9 slides) plus a short backup section
(4 slides), following the paper's argumentation, with speaking narrative in the
speaker notes. Run with the clean_env python (has python-pptx)."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

PAPER = "/home/jlmorg1/work/Johnny_AI_ML_LLM_experiments_2024/papers/ai-metadata-session-data-repository"
OUT_DIR = "/home/jlmorg1/.openclaw/workspace/deck_build"
IMG_DIR = os.path.join(OUT_DIR, "figs")
PPTX = os.path.join(OUT_DIR, "AI_Metadata_Repository_Briefing.pptx")

os.makedirs(IMG_DIR, exist_ok=True)

NAVY = RGBColor(0x11, 0x2A, 0x46)
ACCENT = RGBColor(0x1F, 0x6F, 0x8B)
INK = RGBColor(0x20, 0x24, 0x28)
MUT = RGBColor(0x5A, 0x63, 0x6B)

STANDALONE = r"""\documentclass[border=8pt]{standalone}
\usepackage{times}
\usepackage{tikz}
\usepackage{pgfplots}\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta, positioning, shapes.geometric, fit, backgrounds, matrix, calc, decorations.pathreplacing}
\usepackage{amsmath}
\renewcommand{\ref}[1]{}
\begin{document}
\input{FIG}
\end{document}
"""


def render_figures():
    figs = {
        "fig1": "fig1_architecture_flow.tex",
        "fig2": "fig2_data_model.tex",
        "fig3": "fig3_measurement.tex",
        "fig4": "fig4_lifecycle.tex",
        "fig5": "fig5_deployment.tex",
        "fig6": "fig6_forensics.tex",
    }
    out = {}
    for name, fn in figs.items():
        with open(os.path.join(PAPER, fn), encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r"\s*\(Section~\\ref\{[^}]*\}\)", "", content)
        content = re.sub(r"Section~\\ref\{[^}]*\}", "the linked record", content)
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, f"{name}.tex"), "w", encoding="utf-8") as f:
                f.write(content)
            with open(os.path.join(td, "wrap.tex"), "w", encoding="utf-8") as f:
                f.write(STANDALONE.replace("FIG", name))
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "wrap.tex"],
                           cwd=td, capture_output=True, text=True)
            pdf = os.path.join(td, "wrap.pdf")
            if not os.path.exists(pdf):
                print(f"  [FAIL] {name}")
                continue
            subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", pdf,
                            os.path.join(IMG_DIR, name)], check=True)
            out[name] = os.path.join(IMG_DIR, f"{name}.png")
            print(f"  [ok] {name}")
    return out


def img_aspect(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.width / im.height
    except Exception:
        return 2.5


def build(figs):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    def add_title(slide, text, sub=None, color=NAVY):
        tb = slide.shapes.add_textbox(Inches(0.6), Inches(0.35), Inches(12.1), Inches(1.0))
        tf = tb.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; r = p.add_run(); r.text = text
        r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = color; r.font.name = "Calibri"
        if sub:
            p2 = tf.add_paragraph(); r2 = p2.add_run(); r2.text = sub
            r2.font.size = Pt(15); r2.font.color.rgb = ACCENT; r2.font.name = "Calibri"
        return tb

    def add_bullets(slide, bullets, top, height, width=Inches(12.1), left=Inches(0.7), size=19):
        tb = slide.shapes.add_textbox(left, top, width, height)
        tf = tb.text_frame; tf.word_wrap = True
        first = True
        for text, lvl in bullets:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.level = lvl; p.space_after = Pt(8)
            r = p.add_run(); r.text = ("•  " if lvl == 0 else "–  ") + text
            r.font.size = Pt(size - lvl * 2)
            r.font.color.rgb = INK if lvl == 0 else MUT
            r.font.name = "Calibri"
        return tb

    def add_image(slide, key, top, maxh):
        if not key or key not in figs:
            return
        path = figs[key]; asp = img_aspect(path)
        w = Inches(12.2); h = Inches(w.inches / asp)
        if h > maxh:
            h = maxh; w = Inches(h.inches * asp)
        left = Inches((13.333 - w.inches) / 2)
        slide.shapes.add_picture(path, left, top, width=w, height=h)

    def add_notes(slide, text):
        slide.notes_slide.notes_text_frame.text = text.strip()

    def slide(title, bullets=None, image=None, notes="", sub=None):
        s = prs.slides.add_slide(blank)
        add_title(s, title, sub)
        if image:
            if bullets:
                add_bullets(s, bullets, Inches(1.5), Inches(1.9), size=18)
                add_image(s, image, Inches(3.5), Inches(3.6))
            else:
                add_image(s, image, Inches(1.7), Inches(5.3))
        elif bullets:
            add_bullets(s, bullets, Inches(1.7), Inches(5.3))
        add_notes(s, notes)
        return s

    B = lambda t, l=0: (t, l)

    # ---- TITLE -----------------------------------------------------------
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(2.3), Inches(11.7), Inches(2.8))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]; r = p.add_run(); r.text = "AI Metadata and Session Data Repository"
    r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = NAVY; r.font.name = "Calibri"
    p2 = tf.add_paragraph(); r2 = p2.add_run()
    r2.text = "A Client Architecture for Managing, Measuring, and Mitigating Enterprise Generative AI"
    r2.font.size = Pt(20); r2.font.color.rgb = ACCENT; r2.font.name = "Calibri"
    p3 = tf.add_paragraph(); p3.space_before = Pt(18); r3 = p3.add_run()
    r3.text = "Johnny Morgan  |  UMBC Department of Information Systems"
    r3.font.size = Pt(15); r3.font.color.rgb = MUT; r3.font.name = "Calibri"
    p4 = tf.add_paragraph(); p4.space_before = Pt(6); r4 = p4.add_run()
    r4.text = "Proposed for development and adoption by [Sponsoring Organization], mission sponsor"
    r4.font.size = Pt(13); r4.font.color.rgb = ACCENT; r4.font.name = "Calibri"; r4.font.italic = True
    add_notes(s, """
The argument is one sentence: we cannot manage AI we cannot see, and seeing how this organization uses generative AI requires capturing that use before we know what we will need to ask of it. I am proposing that we build two repositories to hold that record, and that [Sponsoring Organization] sponsor the effort. In the next few minutes: the exposure we carry today, the two-repository architecture I want to build, what it lets us measure and defend, and the ask. This is a proposal and a plan of build, not a report of results.
""")

    # ---- 1 PROBLEM -------------------------------------------------------
    slide("Managing AI we cannot see",
          [B("A model now drafts replies, writes code, summarizes documents, and helps decide, at scale"),
           B("Each interaction is used and then gone: no inventory, no utilization record, no trace of failure"),
           B("We cannot prove the return, and we cannot see the harm, both invisible for the same reason"),
           B("Governance and regulation now treat the record of AI use as an obligation, not an option", 1)],
          notes="""
Adoption is running ahead of management. A model is already drafting clauses, answering customers, and writing code across the organization, and each interaction leaves almost nothing behind. We keep the output and lose the conditions that produced it. That single gap makes two things invisible at once: the value we are getting, which we cannot prove, and the failures we are exposed to, which we cannot see. A manager accountable for an AI deployment cannot today answer basic questions: what do we have deployed, how much do we use it, is it working, how often is it wrong, and does that wrong thing spread. The instinct is to add metrics. That fails, because the failures that matter most are the ones nobody thought to declare in advance. What we lack is not a dashboard, it is a record.
""")

    # ---- 2 PROPOSAL (fig1) ----------------------------------------------
    slide("What we propose: two repositories",
          [B("Content repository: the AI session data itself, preserved as the system of record"),
           B("Metadata repository: everything derived from it, to manage the AI and prove its return"),
           B("Instrumented systems feed the content repository; reporting draws from the metadata repository")],
          image="fig1",
          notes="""
This is the proposal, on the table before I justify it. We build two repositories. The first is a content repository: the AI session data itself, every prompt, turn, tool call, and piece of context, preserved as the organization's system of record. The second is a metadata repository: everything we derive from that content in order to manage the AI, an inventory of what we have deployed, how much it is used, how well it performs, and the return it produces. Instrumented AI systems feed the content repository; the metadata repository is derived from it and segregated by sensitivity; reporting draws from the metadata repository. Everything else in this briefing is the case for why this shape is right and why it has to be built now.
""")

    # ---- 3 METADATA REPOSITORY ------------------------------------------
    slide("The metadata repository: what the investment bought",
          [B("Inventory: which AI systems, models, and versions are deployed, and by whom"),
           B("Utilization: sessions, tokens, and spend, across teams and workflows"),
           B("Performance and effectiveness: task success, rework, acceptance, escalation"),
           B("Return on investment: cost from the record against attributable outcome value, per objective layer"),
           B("Segregated by sensitivity: risk and continuity metadata governed on their own terms", 1)],
          notes="""
The metadata repository is the management vehicle, and it is where the return case lives. It holds four things leadership asks for and cannot get today. An inventory of what AI systems, models, and versions are actually deployed, and by whom. Utilization: how much we use AI, in sessions, tokens, and spend, by team and workflow. Performance and effectiveness: whether the AI is doing useful work, in task success, rework, acceptance, and escalation. And return on investment: the cost we know from the record set against the outcome value we can attribute to AI use, reported at each layer of the objective hierarchy. Inside it, risk and continuity metadata are segregated by sensitivity and governed on their own terms. This is the repository that turns the AI budget from an act of faith into an account.
""")

    # ---- 4 WHY NOW (capture-first + liability) ---------------------------
    slide("Why it must be built now",
          [B("The failures worth measuring are not known in advance; capture must precede the question"),
           B("Anything not captured before it emerges is unrecoverable: a permanent blind spot"),
           B("AI harms are foreseeable, detectable, and mitigatable"),
           B("A harm we could have caught but did not record is one we own; the record makes it a managed event", 1)],
          notes="""
Two reasons this is time-sensitive. First, epistemology. A new failure mode only becomes measurable after enough instances accumulate and someone builds the lens to see it, and both happen after capture. So capture cannot wait on knowing what to look for. And the asymmetry is brutal: capture we omit before a phenomenon emerges cannot be recovered afterward. There is no retrospective study of data nobody wrote down. Capturing now is cheap; reconstructing later is impossible. Second, liability. An AI harm is foreseeable, because a prompt that works today can fail tomorrow; detectable, because the interaction can be examined; and mitigatable, because a caught fault can be corrected before it spreads. An organization that could have foreseen, detected, and mitigated a harm, and did not, because it kept no record, owns the consequence. The record converts a realized loss into a managed event.
""")

    # ---- 5 RISK & FORENSICS (fig6) --------------------------------------
    slide("Managing risk and harm: from detection to forensics",
          [B("Hallucination detection, mitigation, and prevention on a measured adverse-event lifecycle"),
           B("After a harm: reconstruct the trigger, attribute human actor vs AI actor (insider misuse vs rogue action), trace the cascade"),
           B("Prescribe guardrails matched to cause: prompt engineering, review gates, training, permission limits; measure that they worked")],
          image="fig6",
          notes="""
The same record is the risk and safety substrate. Hallucination is not one thing; we classify it against a taxonomy and manage it on an adverse-event lifecycle borrowed from public health: instance, to tracked incidence, to cause and cascade, to mitigation whose efficacy is measured across a dated boundary, to prevention. And when a harm does occur, the harm is not the end of the event, it opens a required investigation. The record lets us reconstruct the trigger and conditions, attribute the event by separating what the human did from what the AI did, which distinguishes insider misuse, where a person directs the AI against us, from rogue or agentic behavior, where the model itself acts unsafely, and trace how far it spread. The investigation prescribes a guardrail matched to the cause, prompt engineering, a human review gate, workforce training, a permission limit, and because each guardrail is dated, we can prove it worked. This covers the surfaces you named: hallucination, insider threat, rogue behavior, refusal, and forensics.
""")

    # ---- 6 RETURN + MANDATE ---------------------------------------------
    slide("Proving return, and meeting the mandate",
          [B("Return is a computed account: cost from the record against attributable outcome value, per layer"),
           B("The question 'how far does AI multiply human capital' becomes measured, not asserted"),
           B("Supplies the record NIST AI RMF, ISO 42001 and 23894, the EU AI Act, and OWASP LLM Top 10 presume"),
           B("Governed by design: segregation by sensitivity, pseudonymous identity, bounded monitoring", 1)],
          notes="""
Two payoffs that matter to a sponsor. First, return becomes an account rather than an anecdote. Cost we know from the record; outcome value we know from the outcome store; return is the relation between them, reported at each layer of the objective hierarchy. The question an executive most wants answered, how far AI multiplies what our people produce, becomes a computed quantity tracked over time. Second, this de-risks the compliance position. The architecture supplies exactly the record that the NIST AI Risk Management Framework, ISO 42001 and 23894, the EU AI Act's logging obligations, and the OWASP LLM Top 10 all presume but do not provide. And governance is built in, not bolted on: stores are segregated by sensitivity, identity is pseudonymous by default so we measure roles and workflows rather than people, and insider-threat monitoring is bounded by purpose, access, and proportionality.
""")

    # ---- 7 SPONSOR & NEXT STEPS -----------------------------------------
    slide("Mission sponsor and next steps",
          [B("Proposed for [Sponsoring Organization] as mission sponsor"),
           B("Organization data to provide: AI systems inventory, objective hierarchy, deployment surfaces"),
           B("Phased build: content repository first, then metadata derivation, then reporting and ROI"),
           B("Pilot on one bounded workflow, measure against its objectives, then widen"),
           B("The paper's dated, falsifiable predictions become the acceptance tests for the pilot", 1)],
          notes="""
Here is the ask and the plan. I am proposing [Sponsoring Organization] as the mission sponsor. What I need from the organization is its own data: the inventory of AI systems in use, the objective hierarchy each layer answers to, and the surfaces where AI is used so we know where to instrument. The build is phased and low-risk: stand up the content repository first, because everything depends on capture; derive the metadata repository from it; then add reporting and the return view. We do not boil the ocean, we pilot on one bounded workflow, measure it against its own objectives, and widen only once it earns that. And the paper states dated, falsifiable predictions that become the pilot's acceptance tests, so success or failure is a matter of record, not opinion.
""")

    # ---- 8 CLOSE ---------------------------------------------------------
    slide("The ask: build the record first",
          [B("On the record, AI harms become foreseeable, detectable, and mitigatable"),
           B("On the same record, AI's value, including how far it multiplies human capital, becomes calculable"),
           B("Keep the record and we can both defend against AI liability and prove its return; without it, neither"),
           B("Capture now is cheap; reconstruction later is impossible. The decision is time-sensitive.", 1)],
          notes="""
To close, back to the one sentence. An organization cannot manage what it cannot see, and it cannot see its use of AI without a record of that use. Build the two repositories that hold that record and two things become true at once: the harms of AI become foreseeable, detectable, and mitigatable, and the value of AI, including how far it multiplies our people's output, becomes calculable. The same record serves both, so an organization that keeps it can both defend against AI liability and prove its return, and one that does not can do neither. The one property I want to leave you with is that this is time-sensitive: capture we do not start today is capture we can never recover. The ask is simple, and it is to sponsor building the record first.
""")

    # ---- BACKUP DIVIDER --------------------------------------------------
    s = prs.slides.add_slide(blank)
    tb = s.shapes.add_textbox(Inches(0.8), Inches(3.1), Inches(11.7), Inches(1.2))
    tf = tb.text_frame
    r = tf.paragraphs[0].add_run(); r.text = "Backup material"
    r.font.size = Pt(34); r.font.bold = True; r.font.color.rgb = MUT; r.font.name = "Calibri"
    p2 = tf.add_paragraph(); r2 = p2.add_run()
    r2.text = "Data model  ·  measurement mechanics  ·  hallucination lifecycle  ·  registered predictions"
    r2.font.size = Pt(15); r2.font.color.rgb = ACCENT; r2.font.name = "Calibri"
    add_notes(s, "Backup slides for likely questions: how capture works, how ROI is computed, the hallucination detail, and the falsifiable predictions.")

    # ---- B1 DATA MODEL (fig2) -------------------------------------------
    slide("Backup: the immutable, linked event",
          [B("Each interaction is one immutable event: the text and the conditions that produced it"),
           B("Captures system prompt, decoding parameters, model version, tool calls, human edits and acceptances"),
           B("Antecedent and consequent links; a raw append-only form and a queryable structured form")],
          image="fig2",
          notes="""
How capture actually works. Each interaction is stored as one immutable event that records the exchanged text and the conditions that produced it: the system prompt, the decoding parameters, the model version, the tool calls, and the human interactions, the edits, acceptances, and rejections. A hallucination produced under a permissive prompt is a different event, with a different cause, from the same words under a constraining one. Events link to their antecedents and consequents, so a cascade is traversed rather than guessed, and each is kept twice: a raw append-only form that is never rewritten, and a structured, queryable form. A later re-classification adds a label; it never alters the original. This is the content repository, and every downstream number and detection is a query over it.
""")

    # ---- B2 MEASUREMENT (fig3) ------------------------------------------
    slide("Backup: how the measures and ROI are computed",
          [B("Objectives are exogenous and layered: workflow, division, corporate, each defines its own measures"),
           B("The architecture computes them over the record and rolls outcomes up the hierarchy"),
           B("MOPs close to the interaction, MOEs against the objective, KPIs business-facing; plus acumen and the trust gap")],
          image="fig3",
          notes="""
How the metadata repository produces its numbers. Measures are not properties of the architecture; they are the organization's objectives, defined in layers, corporate at the top, division and workflow below, each naming its own measures. The architecture does not impose a fixed metric set; it collects, detects, catalogs, and stores the outcomes those measures are computed from and rolls them up. Measures of performance sit close to the interaction, latency, availability, token cost; measures of effectiveness against the objective a workflow serves, task success, rework, escalation; KPIs are the business-facing roll-ups. On top it adds measures a fixed dashboard misses: workforce acumen, model drift, and the gap between how much people trust the model and how reliable it is. Return is cost from the record set against attributable outcome value, per layer.
""")

    # ---- B3 LIFECYCLE + TAXONOMY (fig4) ---------------------------------
    slide("Backup: the hallucination lifecycle and taxonomy",
          [B("Instance, to tracked incidence, to cause and cascade, to mitigation, to prevention"),
           B("Efficacy is a change in incidence across a dated boundary; upstream mitigation becomes prevention"),
           B("Taxonomy: fabrication, overconfidence, staleness, context loss, attribution, scope divergence, agentic failure; refusal is adjacent")],
          image="fig4",
          notes="""
The hallucination detail. We borrow the shape from adverse-event surveillance. A single failure is an instance, and its importance is not evident at the moment it occurs, because impact is a function of trajectory. Repeated instances form an incidence we track. Some trace to a cause; some cascade. A mitigation applied after detection has an efficacy measured as a change in incidence across the dated boundary where we applied it, and a mitigation moved upstream becomes prevention. The taxonomy names the forms so detection knows what to look for: fabrication, overconfidence and misrepresentation, staleness, context loss, attribution error, scope and instruction divergence, and agentic or tool-use failure. Refusal, over- and under-restriction, is tracked alongside as an adjacent mode. Fabrication is the sharpest, and provenance is the defense: it keeps a generated assertion distinguishable from an established fact.
""")

    # ---- B4 PREDICTIONS -------------------------------------------------
    slide("Backup: registered predictions (acceptance tests)",
          [B("Hallucination recurrence clusters around context-window saturation and compaction, not at random"),
           B("Newly detectable failure classes rise with the age and size of a broadly captured corpus"),
           B("Mitigation efficacy is measurable as an incidence change across a dated boundary"),
           B("Without the record, an incident's impact is systematically underestimated"),
           B("Guardrail efficacy is larger upstream of the cause than at the point of harm"),
           B("Over-refusal and under-refusal trade off as the instruction frame tightens")],
          notes="""
Because this is a proposal, its claims are on the record now as dated, falsifiable predictions, and they double as the pilot's acceptance tests. Recurrence should cluster around context saturation and compaction, not arrive at random. In a broadly captured corpus, the rate of discovering new failure classes should rise with the corpus's age and size. Mitigation efficacy should be visible as an incidence change across the dated boundary. Without a repository, an incident's true impact should be systematically underestimated, because the cascade is invisible at the moment it occurs. Guardrails upstream of a cause should outperform guardrails at the point of harm. And tightening a system prompt should trade over-refusal against under-refusal. If the deployed record does not show these, we will know, and that is the point.
""")

    prs.save(PPTX)
    n = len(prs.slides._sldIdLst)
    print(f"\n[deck] {n} slides -> {PPTX}")


if __name__ == "__main__":
    print("Rendering figures...")
    figs = render_figures()
    print("Building deck...")
    build(figs)
