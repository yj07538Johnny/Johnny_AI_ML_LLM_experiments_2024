#!/usr/bin/env python3
"""Figure generator for the information-need classification white paper and deck.

Emits every figure used by:
  docs/information_need_classification/information_need_classification.tex
  docs/information_need_classification/information_need_classification.pptx

Design task: predict which information needs a tagged text satisfies, by running
a DistilBERT-MNLI cross-encoder over (text, need justification) pairs so that
self-attention spans both segments. Multi-label, thousands of candidate needs.

All figures are drawn with matplotlib only (no seaborn, no external assets) so
the script runs anywhere the paper builds. Output is 200 dpi PNG.

Usage:
    python make_figures.py [--out DIR]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

# ---------------------------------------------------------------- style ------

INK = "#111827"
MUTED = "#6b7280"
FAINT = "#d1d5db"
PAPER = "#ffffff"
BLUE = "#2563eb"      # the tagged text side
BLUE_L = "#dbeafe"
AMBER = "#b45309"     # the information-need justification side
AMBER_L = "#fef3c7"
GREEN = "#047857"     # positives / gold
GREEN_L = "#d1fae5"
RED = "#b91c1c"       # negatives
RED_L = "#fee2e2"
VIOLET = "#6d28d9"
VIOLET_L = "#ede9fe"
SLATE_L = "#f1f5f9"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "savefig.facecolor": PAPER,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DPI = 200


def _blank(w, h):
    """A figure with one axes, no ticks, unit coordinates."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, fc=SLATE_L, ec=None, fs=9, tc=INK,
        lw=1.3, weight="normal", pad=0.4, z=2):
    ec = ec if ec is not None else MUTED
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=z + 1, linespacing=1.4,
            fontweight=weight)


def arrow(ax, p0, p1, color=MUTED, lw=1.5, rad=0.0, ls="-", style="-|>", z=3):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw,
                                linestyle=ls, shrinkA=1, shrinkB=1,
                                connectionstyle=f"arc3,rad={rad}"))


def title(ax, text, sub=None):
    ax.text(50, 96, text, ha="center", va="top", fontsize=12.5,
            fontweight="bold", color=INK)
    if sub:
        ax.text(50, 89.5, sub, ha="center", va="top", fontsize=8.8, color=MUTED)


def caption(ax, text, y=2.5):
    ax.text(50, y, text, ha="center", va="bottom", fontsize=8, color=MUTED,
            style="italic")


# ------------------------------------------------------- fig 1: sup. loop ----

def fig_supervised_loop(out):
    fig, ax = _blank(9.6, 5.0)
    title(ax, "Supervised learning, one training step",
          "The loop repeats over mini-batches until the loss stops falling on held-out data")

    box(ax, 3, 52, 20, 20,
        "Labelled data\n$(x_i,\\ y_i)$\n\na report paired with\nits cited needs",
        fc=SLATE_L, fs=8.6)
    box(ax, 30, 52, 20, 20, "Model $f_\\theta$\n\nDistilBERT-MNLI\ncross-encoder",
        fc=BLUE_L, ec=BLUE, fs=8.6)
    box(ax, 57, 52, 17, 20, "Prediction\n$\\hat{y} = f_\\theta(x)$", fc=SLATE_L, fs=8.6)
    box(ax, 79, 52, 18, 20, "Loss\n$L(\\hat{y},\\ y)$\n\nhow wrong,\nas one number",
        fc=RED_L, ec=RED, fs=8.6)
    box(ax, 30, 17, 44, 16,
        "Update  $\\theta \\leftarrow \\theta - \\eta\\, \\nabla_\\theta L$\n"
        "gradient descent moves every weight a little\nin the direction that lowers the loss",
        fc=GREEN_L, ec=GREEN, fs=8.6)

    arrow(ax, (23.5, 62), (29.5, 62))
    arrow(ax, (50.5, 62), (56.5, 62))
    arrow(ax, (74.5, 62), (78.5, 62))
    arrow(ax, (88, 51.5), (74.5, 33.5), color=GREEN, rad=-0.22, lw=1.8)
    arrow(ax, (30, 25), (13, 25), color=GREEN, lw=1.8)
    arrow(ax, (13, 25), (13, 51.5), color=GREEN, lw=1.8, style="-")
    arrow(ax, (13, 45), (13, 51.5), color=GREEN, lw=1.8)
    arrow(ax, (40, 51.5), (40, 33.5), color=GREEN, ls=":", lw=1.5, style="<|-")

    ax.text(84, 41, "backward pass", fontsize=8, color=GREEN, rotation=-34,
            ha="center", style="italic")
    ax.text(21.5, 28.5, "next batch", fontsize=8, color=GREEN, ha="center",
            va="bottom", style="italic")
    ax.text(43, 42, "weights\nchange", fontsize=8, color=GREEN, ha="left",
            va="center", style="italic")

    caption(ax, "Nothing here is specific to transformers. The model in the blue box is the only part we swap.")
    fig.tight_layout()
    fig.savefig(out / "fig01_supervised_loop.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- fig 2: attention math -----

def fig_attention_mechanism(out):
    fig, ax = _blank(9.6, 5.6)
    title(ax, "Self-attention: how one token reads the others",
          "Every token emits a Query, a Key and a Value. Queries are matched against Keys to weight the Values.")

    toks = ["coastal", "cities", "raised", "zoning"]
    for i, t in enumerate(toks):
        box(ax, 4 + i * 12, 71, 10.5, 9, t, fc=SLATE_L, fs=8.4)

    for i in range(4):
        arrow(ax, (9.2 + i * 12, 70.5), (9.2 + i * 12, 62.5), lw=1.1)

    for j, (nm, col, coll) in enumerate([("Q", BLUE, BLUE_L),
                                         ("K", AMBER, AMBER_L),
                                         ("V", VIOLET, VIOLET_L)]):
        for i in range(4):
            box(ax, 4 + i * 12 + j * 0.0, 53 - j * 0.0, 10.5, 8.5,
                "", fc=coll, ec=col, fs=8, z=2 - j)
        ax.text(2.5, 57.2, nm, fontsize=11, fontweight="bold", color=col,
                ha="right", va="center")

    for i, t in enumerate(toks):
        ax.text(9.2 + i * 12, 57.2, t[:4], ha="center", va="center",
                fontsize=7.6, color=INK)

    ax.text(56, 57.2, "$W_Q,\\ W_K,\\ W_V$ are learned projections",
            fontsize=8.5, color=MUTED, ha="left", va="center")

    box(ax, 4, 30, 92, 15,
        "$\\mathrm{Attention}(Q,K,V) \\;=\\; \\mathrm{softmax}\\!\\left("
        "\\dfrac{Q K^{\\top}}{\\sqrt{d_k}}\\right) V$",
        fc=PAPER, ec=INK, fs=15, lw=1.6)

    ax.text(15, 24, "$QK^{\\top}$\nevery token scored\nagainst every token",
            ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.text(42, 24, "$\\sqrt{d_k}$\nkeeps the scores from\nsaturating the softmax",
            ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.text(70, 24, "softmax\nturns scores into\nweights summing to 1",
            ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.text(92, 24, "$\\cdot V$\nweighted blend\nof the values",
            ha="center", va="top", fontsize=8.2, color=MUTED)

    caption(ax, "One head. DistilBERT runs 12 heads per layer and concatenates them, then repeats for 6 layers.")
    fig.tight_layout()
    fig.savefig(out / "fig02_attention_mechanism.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------ fig 3: cross-segment attention map -----

def _synthetic_attention():
    text_toks = ["[CLS]", "coastal", "cities", "raised", "zoning", "limits", "[SEP]"]
    just_toks = ["municipal", "response", "to", "sea", "level", "rise", "[SEP]"]
    toks = text_toks + just_toks
    n = len(toks)
    nt = len(text_toks)

    rng = np.random.default_rng(11)
    a = rng.uniform(0.05, 0.25, size=(n, n))
    a += np.eye(n) * 0.45

    def bump(i, j, v):
        a[i, j] += v
        a[j, i] += v * 0.85

    # the semantic bridges that make this pair entail
    bump(toks.index("coastal"), nt + just_toks.index("sea"), 1.5)
    bump(toks.index("coastal"), nt + just_toks.index("level"), 1.1)
    bump(toks.index("cities"), nt + just_toks.index("municipal"), 1.9)
    bump(toks.index("zoning"), nt + just_toks.index("response"), 1.4)
    bump(toks.index("limits"), nt + just_toks.index("response"), 1.0)
    bump(toks.index("raised"), nt + just_toks.index("rise"), 1.2)

    a[0, :] += 0.55                      # [CLS] reads broadly
    a[:, 0] += 0.15
    a = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
    return toks, nt, a


def fig_cross_segment_attention(out):
    toks, nt, a = _synthetic_attention()
    n = len(toks)

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    im = ax.imshow(a, cmap="Blues", vmin=0, vmax=a.max())

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(toks, rotation=55, ha="right", fontsize=8)
    ax.set_yticklabels(toks, fontsize=8)
    for lbl, i in zip(ax.get_xticklabels(), range(n)):
        lbl.set_color(BLUE if i < nt else AMBER)
    for lbl, i in zip(ax.get_yticklabels(), range(n)):
        lbl.set_color(BLUE if i < nt else AMBER)

    ax.axhline(nt - 0.5, color=INK, lw=1.8)
    ax.axvline(nt - 0.5, color=INK, lw=1.8)

    for (x0, y0, w, h, col, lab) in [
        (-0.5, -0.5, nt, nt, BLUE, "text\nreads text"),
        (nt - 0.5, nt - 0.5, n - nt, n - nt, AMBER, "need\nreads need"),
    ]:
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=col,
                               lw=2.0, zorder=4))
        ax.text(x0 + w / 2, y0 + h / 2, lab, ha="center", va="center",
                fontsize=8.5, color=col, alpha=0.55, fontweight="bold", zorder=5)

    for (x0, y0, w, h) in [(nt - 0.5, -0.5, n - nt, nt),
                           (-0.5, nt - 0.5, nt, n - nt)]:
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=RED,
                               lw=2.4, linestyle=(0, (4, 2)), zorder=4))

    ax.text(nt + (n - nt) / 2 - 0.5, nt / 2 - 0.5,
            "PAIRWISE\nATTENTION", ha="center", va="center", fontsize=10,
            color=RED, fontweight="bold", zorder=6)

    ax.set_title("Attention inside a cross-encoder\n"
                 "One sequence: [CLS] report [SEP] need justification [SEP]",
                 fontsize=11.5, fontweight="bold", pad=14)
    ax.set_xlabel("attended to (key)", fontsize=9)
    ax.set_ylabel("attending from (query)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.62, label="attention weight")

    fig.text(0.5, 0.005,
             "The dashed blocks are the only reason this architecture beats two separate encoders.\n"
             "\"cities\" reads \"municipal\", \"raised\" reads \"rise\", inside the network rather than after it.",
             ha="center", fontsize=8, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(out / "fig03_cross_segment_attention.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------- fig 4: cross vs bi-encoder ----------

def fig_encoder_comparison(out):
    fig, ax = _blank(10.2, 5.4)
    title(ax, "Why a cross-encoder, and what it costs",
          "The two designs differ only in where the text and the justification are allowed to meet")

    ax.text(25, 84, "CROSS-ENCODER  (what we train)", ha="center",
            fontsize=9.5, fontweight="bold", color=GREEN)
    ax.text(75, 84, "BI-ENCODER  (the fast alternative)", ha="center",
            fontsize=9.5, fontweight="bold", color=MUTED)
    ax.plot([50, 50], [8, 80], color=FAINT, lw=1.2, ls="--")

    # left: cross encoder
    box(ax, 4, 68, 20, 9, "report", fc=BLUE_L, ec=BLUE, fs=8.4)
    box(ax, 26, 68, 20, 9, "justification", fc=AMBER_L, ec=AMBER, fs=8.4)
    box(ax, 4, 55, 42, 8.5, "concatenate  [CLS] · [SEP] · [SEP]", fc=SLATE_L, fs=8.4)
    box(ax, 4, 36, 42, 15,
        "DistilBERT, 6 layers\n\nattention spans BOTH segments\nat every layer",
        fc=GREEN_L, ec=GREEN, fs=8.6)
    box(ax, 12, 22, 26, 9, "[CLS] $\\rightarrow$ entailment score", fc=SLATE_L, fs=8.4)
    arrow(ax, (14, 67.5), (14, 64))
    arrow(ax, (36, 67.5), (36, 64))
    arrow(ax, (25, 54.5), (25, 51.5))
    arrow(ax, (25, 35.5), (25, 31.5))
    ax.text(25, 15, "one forward pass per PAIR", ha="center", fontsize=8.4,
            color=GREEN, fontweight="bold")
    ax.text(25, 10.5, "cannot cache the need side", ha="center", fontsize=8, color=RED)

    # right: bi encoder
    box(ax, 54, 68, 18, 9, "report", fc=BLUE_L, ec=BLUE, fs=8.4)
    box(ax, 78, 68, 18, 9, "justification", fc=AMBER_L, ec=AMBER, fs=8.4)
    box(ax, 54, 47, 18, 15, "DistilBERT", fc=SLATE_L, fs=8.4)
    box(ax, 78, 47, 18, 15, "DistilBERT", fc=SLATE_L, fs=8.4)
    box(ax, 54, 34, 18, 8, "$u$  (768-d)", fc=BLUE_L, ec=BLUE, fs=8.4)
    box(ax, 78, 34, 18, 8, "$v$  (768-d)", fc=AMBER_L, ec=AMBER, fs=8.4)
    box(ax, 62, 21, 26, 8.5, "$\\cos(u,\\ v)$", fc=SLATE_L, fs=9)
    for x in (63, 87):
        arrow(ax, (x, 67.5), (x, 62.5))
        arrow(ax, (x, 46.5), (x, 42.5))
    arrow(ax, (63, 33.5), (70, 30), rad=0.15)
    arrow(ax, (87, 33.5), (80, 30), rad=-0.15)
    ax.text(75, 15, "needs embedded ONCE, cached", ha="center", fontsize=8.4,
            color=GREEN, fontweight="bold")
    ax.text(75, 10.5, "the two sides never attend to each other", ha="center",
            fontsize=8, color=RED)

    caption(ax, "We use both: the bi-encoder narrows thousands of needs to a shortlist, the cross-encoder ranks the shortlist.")
    fig.tight_layout()
    fig.savefig(out / "fig04_encoder_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- fig 5: distilbert-mnli ----------

def fig_distilbert_mnli(out):
    fig, ax = _blank(10.2, 4.6)
    title(ax, "Where DistilBERT-MNLI comes from, and what we change",
          "Three rounds of training already happened before we start. We only re-fit the last box.")

    stages = [
        (2, "BERT-base\n\n12 layers\n110M params",
         "masked language\nmodelling\non raw text", SLATE_L, MUTED),
        (26, "DistilBERT\n\n6 layers\n66M params",
         "knowledge distillation\n40% smaller,\n60% faster", SLATE_L, MUTED),
        (50, "DistilBERT-MNLI\n\n+ 3-way head",
         "fine-tuned on 393k\npremise/hypothesis\npairs", VIOLET_L, VIOLET),
        (74, "Our model\n\n+ 1 sigmoid",
         "fine-tuned on\n(text, justification)\npairs", GREEN_L, GREEN),
    ]
    for x, head, sub, fc, ec in stages:
        box(ax, x, 46, 22, 26, head, fc=fc, ec=ec, fs=9)
        ax.text(x + 11, 40, sub, ha="center", va="top", fontsize=8, color=MUTED)

    for x in (24, 48, 72):
        arrow(ax, (x, 59), (x + 2, 59), lw=1.8, color=INK)

    box(ax, 50, 8, 46, 20,
        "Why MNLI transfers:\n"
        "MNLI already asks \"does the premise support the hypothesis?\"\n"
        "We are asking \"does this text satisfy this information need?\"\n"
        "Same shape of question, different vocabulary.",
        fc=VIOLET_L, ec=VIOLET, fs=8.4)

    box(ax, 2, 8, 44, 20,
        "What we replace:\n"
        "MNLI emits entail / neutral / contradict.\n"
        "We keep the encoder, discard the 3-way head, and\n"
        "attach one sigmoid output per pair (multi-label).",
        fc=GREEN_L, ec=GREEN, fs=8.4)

    fig.tight_layout()
    fig.savefig(out / "fig05_distilbert_mnli.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- fig 6: pair construction --------

def fig_pair_construction(out):
    fig, ax = _blank(10.2, 5.6)
    title(ax, "Turning two datasets into training pairs",
          "The label is not on the text. It is on the PAIR. This is the step that defines the task.")

    box(ax, 3, 60, 26, 24,
        "Reports\n\n$r_1 \\ldots r_N$\n\n$N \\approx 20{,}000$",
        fc=BLUE_L, ec=BLUE, fs=8.6)
    box(ax, 71, 60, 26, 24,
        "Need repository\n\n$n_1 \\ldots n_M$\n\n$M > 1{,}000$, each with\na justification",
        fc=AMBER_L, ec=AMBER, fs=8.6)
    box(ax, 34, 62, 32, 20,
        "Cross product\n$N \\times M \\approx 20$ million\n\nfar too many to score",
        fc=SLATE_L, fs=8.6)
    arrow(ax, (29.5, 72), (33.5, 72))
    arrow(ax, (70.5, 72), (66.5, 72), style="-|>")

    box(ax, 3, 26, 28, 22,
        "POSITIVES\n\n$(r_i,\\ n_j)$ where an author\ncited $r_i$ against $n_j$\n\ntarget = 1",
        fc=GREEN_L, ec=GREEN, fs=8.6)
    box(ax, 36, 26, 28, 22,
        "HARD NEGATIVES\n\nneeds the bi-encoder ranks\nhighly but that are NOT gold\n\ntarget = 0",
        fc=RED_L, ec=RED, fs=8.6)
    box(ax, 69, 26, 28, 22,
        "RANDOM NEGATIVES\n\nuniformly sampled needs\n\ntarget = 0",
        fc=SLATE_L, ec=MUTED, fs=8.6)

    arrow(ax, (44, 61.5), (17, 48.5), rad=0.16, color=GREEN)
    arrow(ax, (50, 61.5), (50, 48.5), color=RED)
    arrow(ax, (56, 61.5), (83, 48.5), rad=-0.16)

    box(ax, 12, 5, 76, 14,
        "Sampling ratio per positive:  1 gold  :  4 hard negatives  :  4 random negatives\n"
        "Hard negatives teach the boundary. Random negatives stop the model calling everything relevant.",
        fc=PAPER, ec=INK, fs=8.6)

    fig.tight_layout()
    fig.savefig(out / "fig06_pair_construction.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- fig 7: training pipeline --------

def fig_training_pipeline(out):
    fig, ax = _blank(10.6, 5.0)
    title(ax, "The training pipeline, end to end",
          "One mini-batch of pairs, from raw strings to a weight update")

    steps = [
        ("1. Sample\n\na batch of pairs\nwith their targets", SLATE_L, MUTED),
        ("2. Tokenise\n\ntext_pair encoding,\ntruncate to 256", SLATE_L, MUTED),
        ("3. Encode\n\n6 transformer layers,\nattention spans both", BLUE_L, BLUE),
        ("4. Pool\n\ntake the [CLS]\nvector, 768-d", SLATE_L, MUTED),
        ("5. Score\n\nLinear + sigmoid\n$\\rightarrow p \\in (0,1)$", VIOLET_L, VIOLET),
        ("6. Loss\n\nbinary cross-entropy\nagainst the target", RED_L, RED),
        ("7. Update\n\nAdamW, lr 2e-5,\nlinear warmup", GREEN_L, GREEN),
    ]
    w, gap = 12.2, 1.9
    x = 2.0
    for label, fc, ec in steps:
        box(ax, x, 40, w, 30, label, fc=fc, ec=ec, fs=8.0)
        if x > 2.0:
            arrow(ax, (x - gap - 0.3, 55), (x - 0.4, 55), lw=1.4)
        x += w + gap

    arrow(ax, (92, 39.5), (92, 27), color=GREEN, lw=1.6)
    arrow(ax, (92, 27), (8, 27), color=GREEN, lw=1.6, style="-")
    arrow(ax, (20, 27), (8, 27), color=GREEN, lw=1.6)
    arrow(ax, (8, 27), (8, 39.5), color=GREEN, lw=1.6)
    ax.text(50, 23.5, "repeat for every batch, 3 epochs", ha="center",
            fontsize=8.4, color=GREEN, style="italic")

    box(ax, 6, 5, 88, 13,
        "Held-out evaluation after each epoch. Stop when validation mAP stops improving, not when training loss stops falling.\n"
        "The two diverge, and the gap between them is the only early warning of overfitting we get.",
        fc=PAPER, ec=INK, fs=8.4)

    fig.tight_layout()
    fig.savefig(out / "fig07_training_pipeline.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- fig 8: multi-label eval ---------

def fig_multilabel_eval(out):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    # left: precision-recall with threshold sweep
    ax = axes[0]
    rec = np.linspace(0.02, 1.0, 250)
    prec = np.clip(0.97 - 0.72 * rec ** 2.1, 0.06, 1.0)
    ax.plot(rec, prec, color=BLUE, lw=2.2)
    ax.fill_between(rec, 0, prec, color=BLUE_L, alpha=0.55)

    # Label positions are absolute, parked in the empty wedge above the curve, so
    # they collide with neither the axes title nor each other.
    for r, lab, col, tx, ty, ha in [
            (0.30, "$\\tau=0.8$   precise, misses a lot", AMBER, 0.40, 0.99, "left"),
            (0.63, "$\\tau=0.5$   balanced", GREEN, 0.66, 0.84, "left"),
            (0.90, "$\\tau=0.2$   noisy", RED, 0.99, 0.60, "right")]:
        p = np.interp(r, rec, prec)
        ax.plot([r], [p], "o", color=col, ms=8, zorder=5)
        ax.annotate(lab, xy=(r, p), xytext=(tx, ty),
                    fontsize=8, color=col, ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=1, alpha=0.7))
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.08)
    ax.set_title("The threshold is a product decision, not a metric",
                 fontsize=10, fontweight="bold")

    # right: per-label support vs F1
    ax = axes[1]
    rng = np.random.default_rng(7)
    support = np.concatenate([
        rng.integers(1, 8, 140), rng.integers(8, 60, 70), rng.integers(60, 900, 26)])
    f1 = np.clip(0.22 + 0.29 * np.log10(support + 1) + rng.normal(0, 0.075, support.size), 0.02, 0.97)
    ax.scatter(support, f1, s=17, color=VIOLET, alpha=0.55, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel("training pairs for that need (log scale)")
    ax.set_ylabel("per-need F1")
    ax.set_ylim(0, 1.0)
    ax.axhline(np.mean(f1), color=MUTED, ls="--", lw=1.2)
    ax.text(1.15, np.mean(f1) + 0.03, f"macro-F1 = {np.mean(f1):.2f}",
            fontsize=8, color=MUTED)
    ax.set_title("Rare needs are where the model actually fails",
                 fontsize=10, fontweight="bold")

    fig.text(0.5, 0.005,
             "Illustrative shapes, not measured results. Report micro-F1, macro-F1 and mAP together: "
             "micro hides the long tail, macro exposes it.",
             ha="center", fontsize=8, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "fig08_multilabel_eval.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- fig 9: serving ------------------

def fig_serving(out):
    fig, ax = _blank(10.2, 4.6)
    title(ax, "Serving against 1,000+ needs",
          "Exhaustive pair scoring is not affordable. Retrieve first, then attend.")

    box(ax, 2, 55, 17, 22, "incoming\nreport", fc=BLUE_L, ec=BLUE, fs=8.6)
    box(ax, 23, 55, 22, 22,
        "STAGE 1\nbi-encoder retrieval\n\nneeds pre-embedded\nand cached",
        fc=SLATE_L, ec=MUTED, fs=8.4)
    box(ax, 49, 55, 20, 22, "top-$k$ shortlist\n\n$k \\approx 50$",
        fc=AMBER_L, ec=AMBER, fs=8.6)
    box(ax, 73, 55, 24, 22,
        "STAGE 2\ncross-encoder rerank\n\nfull pairwise attention\non $k$ pairs only",
        fc=GREEN_L, ec=GREEN, fs=8.4)

    arrow(ax, (19.5, 66), (22.5, 66))
    arrow(ax, (45.5, 66), (48.5, 66))
    arrow(ax, (69.5, 66), (72.5, 66))

    ax.text(34, 50, "$O(1)$ transformer passes\n+ one vector search",
            ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(85, 50, "$O(k)$ transformer passes\n$k$ is fixed, $M$ is not",
            ha="center", va="top", fontsize=8, color=MUTED)

    box(ax, 8, 8, 84, 26,
        "Cost, per incoming report, with $M = 1{,}000$ needs at ~5 ms per pair:\n\n"
        "exhaustive cross-encoder     1,000 forward passes       ~5 s per report\n"
        "retrieve then rerank              1 + 50 passes                  ~0.25 s per report,\n"
        "                                                                                   and flat as the repository grows",
        fc=PAPER, ec=INK, fs=8.6)

    caption(ax, "Stage 1 sets a hard recall ceiling. Measure recall@k before tuning anything in stage 2.", y=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig09_serving.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- fig 10: worked example ----------

def fig_worked_example(out):
    fig, ax = _blank(10.2, 5.2)
    title(ax, "One text, three candidate needs",
          "The same tagged text is scored against each justification independently")

    box(ax, 6, 74, 88, 16,
        "REPORT\n\"Coastal cities raised zoning limits for new construction within the "
        "surge boundary,\nfunded through a municipal bond approved last spring.\"",
        fc=BLUE_L, ec=BLUE, fs=8.8)

    rows = [
        ("IN-042  Regional climate adaptation planning",
         "Analysts need documents describing municipal responses to sea level rise,\n"
         "including zoning changes and funding mechanisms.",
         0.94, GREEN, GREEN_L, "gold"),
        ("IN-117  Municipal bond issuance",
         "Analysts need documents recording new municipal debt instruments,\n"
         "their size, and their approval process.",
         0.71, GREEN, GREEN_L, "gold"),
        ("IN-233  Coastal shipping capacity",
         "Analysts need documents on port throughput, berth availability,\n"
         "and container volumes at coastal terminals.",
         0.08, RED, RED_L, "not gold"),
    ]
    y = 50
    for name, just, score, col, coll, tag in rows:
        box(ax, 6, y, 62, 18, "", fc=AMBER_L, ec=AMBER, fs=8, pad=0.3)
        ax.text(8.5, y + 14, name, fontsize=8.6, fontweight="bold", color=AMBER,
                va="center", ha="left")
        ax.text(8.5, y + 6.5, just, fontsize=7.8, color=INK, va="center",
                ha="left", linespacing=1.35)
        box(ax, 72, y + 3, 12, 12, f"{score:.2f}", fc=coll, ec=col, fs=12,
            weight="bold", tc=col)
        ax.text(88, y + 9, tag, fontsize=8.2, color=col, ha="center",
                va="center", style="italic")
        arrow(ax, (68.5, y + 9), (71.5, y + 9), color=col, lw=1.4)
        y -= 22

    ax.text(78, 45, "sigmoid\noutput", fontsize=7.6, color=MUTED, ha="center",
            va="bottom")
    box(ax, 6, 2, 88, 9,
        "Multi-label means these three decisions are independent. Two needs fire, one does not, and no softmax forces them to compete.",
        fc=PAPER, ec=INK, fs=8.4)

    fig.tight_layout()
    fig.savefig(out / "fig10_worked_example.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main -------

def fig_citation_label_structure(out):
    """The PU structure: cited = positive, uncited = UNKNOWN, not negative."""
    n_r, n_n = 14, 20
    rng = np.random.default_rng(5)
    cited = np.zeros((n_r, n_n), dtype=int)
    for i in range(n_r):
        for j in rng.choice(n_n, size=rng.integers(1, 4), replace=False):
            cited[i, j] = 1
    # A handful of uncited-but-relevant cells: the discovery target.
    latent = np.zeros_like(cited)
    zeros = np.argwhere(cited == 0)
    for i, j in zeros[rng.choice(len(zeros), size=9, replace=False)]:
        latent[i, j] = 1

    fig = plt.figure(figsize=(11.2, 5.0))
    # Explicit axes placement: the grid must stay square-celled (aspect equal),
    # so it cannot share a tight_layout pass with the legend column.
    ax = fig.add_axes([0.035, 0.16, 0.52, 0.66])
    ax.set_xlim(-0.5, n_n - 0.5)
    ax.set_ylim(n_r - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    for i in range(n_r):
        for j in range(n_n):
            if cited[i, j]:
                fc, ec, lw = GREEN, GREEN, 0.8
            elif latent[i, j]:
                fc, ec, lw = "#ffffff", VIOLET, 1.6
            else:
                fc, ec, lw = SLATE_L, "#e2e8f0", 0.8
            ax.add_patch(Rectangle((j - 0.44, i - 0.44), 0.88, 0.88,
                                   facecolor=fc, edgecolor=ec, linewidth=lw))

    ax.text(0.5, 1.06, "rows = reports          columns = information needs",
            transform=ax.transAxes, ha="center", fontsize=8.5, color=MUTED)

    lg = fig.add_axes([0.60, 0.10, 0.38, 0.74])
    lg.set_xlim(0, 1), lg.set_ylim(0, 1)
    lg.axis("off")
    handles = [
        (GREEN, GREEN, "CITED",
         "An author said this report satisfies\nthis need. A reliable positive."),
        ("#ffffff", VIOLET, "UNCITED BUT RELEVANT",
         "Your data cannot tell these apart from\nthe category below. They are drawn\n"
         "differently here only to make the point.\nThis is the discovery target."),
        (SLATE_L, "#e2e8f0", "UNCITED",
         "Almost all of these are genuinely\nirrelevant, but nothing in the data\nsays which."),
    ]
    y = 0.93
    for fc, ec, head, body in handles:
        lg.add_patch(Rectangle((0.0, y - 0.045), 0.052, 0.055,
                               facecolor=fc, edgecolor=ec, linewidth=1.6))
        lg.text(0.085, y - 0.016, head, fontsize=9.5, fontweight="bold",
                va="center", color=ec if fc == "#ffffff" else INK)
        lg.text(0.085, y - 0.14, body, fontsize=8.4, va="center", color=MUTED,
                linespacing=1.5)
        y -= 0.32

    fig.text(0.5, 0.955, "What a citation does and does not tell you",
             ha="center", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.035,
             "Absence of a citation is NOT a negative label. Treating it as one trains the model "
             "to suppress exactly what you built it to find.",
             ha="center", fontsize=9, color=RED, style="italic", fontweight="bold")
    fig.savefig(out / "fig11_citation_label_structure.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


def fig_data_structure(out):
    fig, ax = _blank(10.6, 6.0)
    title(ax, "How to structure the training data",
          "Three tables you curate, one table the trainer derives")

    box(ax, 2, 68, 28, 17,
        "reports\n\nreport_id      PK\ntext / abstract\npub_date\nauthor_id",
        fc=BLUE_L, ec=BLUE, fs=8.4)
    box(ax, 36, 68, 28, 17,
        "needs\n\nneed_id        PK\nneed_label\njustification\nstatus, opened_on",
        fc=AMBER_L, ec=AMBER, fs=8.4)
    box(ax, 70, 68, 28, 17,
        "citations\n\nreport_id      FK\nneed_id        FK\ncited_by, cited_on\n"
        "confidence",
        fc=GREEN_L, ec=GREEN, fs=8.4)

    ax.text(50, 65.5,
            "the citations table IS your label set: one row = one observed positive edge",
            ha="center", va="top", fontsize=9, color=MUTED, style="italic")

    for x in (16, 50, 84):
        arrow(ax, (x, 67.5), (x, 62), lw=1.4)

    box(ax, 8, 40, 84, 21,
        "pairs   (derived, never hand-edited)\n\n"
        "report_id | need_id | label | source        | split\n"
        "----------|---------|-------|---------------|-------\n"
        "R-00412   | IN-0042 |   1   | cited         | train\n"
        "R-00412   | IN-0311 |   0   | hard_negative | train\n"
        "R-00412   | IN-0908 |   0   | random_negative | train",
        fc=PAPER, ec=INK, fs=8.6)

    box(ax, 8, 20, 40, 15,
        "SPLIT BY REPORT\n\nEvery pair for a report lands in the\nsame split. Splitting pairs "
        "instead\nleaks the report across the boundary.",
        fc=SLATE_L, ec=MUTED, fs=8.4)
    box(ax, 52, 20, 40, 15,
        "ALSO SPLIT BY TIME\n\nHold out the most recent months.\nNeeds are standing "
        "requirements;\nyou will be scoring future reports.",
        fc=SLATE_L, ec=MUTED, fs=8.4)

    box(ax, 8, 3, 84, 13,
        "At your scale: 20,000 reports x 1,000 needs = 20,000,000 candidate pairs.\n"
        "~30,000 are cited (0.15% density). Sampling 8 negatives per positive gives ~270,000 training pairs.",
        fc=VIOLET_L, ec=VIOLET, fs=8.8)

    fig.tight_layout()
    fig.savefig(out / "fig12_data_structure.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_discovery_loop(out):
    fig, ax = _blank(10.6, 5.2)
    title(ax, "The discovery loop",
          "The uncited pairs the model scores highly are the output, not the error")

    box(ax, 2, 62, 21, 22, "trained\ncross-encoder", fc=GREEN_L, ec=GREEN, fs=9)
    box(ax, 27, 62, 23, 22,
        "score the UNCITED\npairs\n\nthe 19,970,000 cells\nwith no citation",
        fc=SLATE_L, fs=8.4)
    box(ax, 54, 62, 21, 22,
        "rank by score\n\ntake the top few\nhundred", fc=AMBER_L, ec=AMBER, fs=8.6)
    box(ax, 79, 62, 19, 22,
        "human\nadjudication\n\nis this real?", fc=VIOLET_L, ec=VIOLET, fs=8.6)

    for x0, x1 in ((23.5, 26.5), (50.5, 53.5), (75.5, 78.5)):
        arrow(ax, (x0, 73), (x1, 73))

    # Outcome boxes sit under the adjudication box so the fork stays short and
    # neither branch crosses the other box.
    box(ax, 50, 31, 23.5, 21,
        "NO, a false positive\n\nA CONFIRMED negative,\nwhich the raw data never\n"
        "gave you. Worth more\nthan a sampled one.",
        fc=RED_L, ec=RED, fs=8.4)
    box(ax, 76, 31, 22, 21,
        "YES, a real relationship\n\nA discovery. Add it to the\ncitations table as a new\n"
        "positive edge and it trains\nthe next round.",
        fc=GREEN_L, ec=GREEN, fs=8.4)

    arrow(ax, (88, 61.5), (87, 52.5), color=GREEN, lw=1.6)
    arrow(ax, (85, 61.5), (63, 52.5), color=RED, lw=1.6, rad=0.18)

    # Both outcomes feed the next training round.
    arrow(ax, (61, 30.5), (61, 23), color=MUTED, lw=1.4)
    arrow(ax, (87, 30.5), (87, 23), color=MUTED, lw=1.4)
    arrow(ax, (87, 23), (12, 23), color=MUTED, lw=1.4, style="-")
    arrow(ax, (30, 23), (12, 23), color=MUTED, lw=1.4)
    arrow(ax, (12, 23), (12, 61.5), color=MUTED, lw=1.4)

    ax.text(8.5, 43, "retrain", fontsize=8.5, color=MUTED, rotation=90,
            ha="center", va="center", style="italic")

    box(ax, 20, 3, 60, 11,
        "Every adjudicated pair is a label you did not have before.\n"
        "This is the only way the unknown region ever shrinks.",
        fc=PAPER, ec=INK, fs=9)

    fig.tight_layout()
    fig.savefig(out / "fig13_discovery_loop.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


FIGURES = [
    fig_supervised_loop,
    fig_attention_mechanism,
    fig_cross_segment_attention,
    fig_encoder_comparison,
    fig_distilbert_mnli,
    fig_pair_construction,
    fig_training_pipeline,
    fig_multilabel_eval,
    fig_serving,
    fig_worked_example,
    fig_citation_label_structure,
    fig_data_structure,
    fig_discovery_loop,
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default=str(Path(__file__).resolve().parent / "figures"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for fn in FIGURES:
        fn(out)
        print(f"  ok  {fn.__name__}")

    made = sorted(out.glob("*.png"))
    print(f"\n{len(made)} figures -> {out}")
    for p in made:
        print(f"  {p.name:44s} {p.stat().st_size / 1024:7.1f} KB")


if __name__ == "__main__":
    main()
