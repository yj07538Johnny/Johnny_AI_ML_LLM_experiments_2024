#!/usr/bin/env python3
"""Two figures for the extended hallucination briefing.

These illustrate observations already formalised in the project's theory
registry. Neither invents a claim; each draws a theory the researcher wrote.

  fig5  Recovery cost by detection depth          T146 (HCR)
  fig6  The consequence asymmetry                 T143 (SDCA) + T144 (SCB)

The extended deck also reuses figures produced elsewhere:
  fig2_eight_classes, fig3_type_breakout          make_figures.py (this dir)
  fig2_fidelity_gradient                          compaction_briefing/

Usage:
    python make_observation_figures.py [--out DIR]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")

import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402

from figkit import (AMBER, AMBER_L, BLUE, BLUE_L, DPI, FAINT,  # noqa: E402
                    GREEN, GREEN_L, INK, MUTED, PAPER, RED, RED_L,
                    SLATE_L, VIOLET, VIOLET_L,
                    arrow, blank, box, caption, figure_cli, title)

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------- fig 5: recovery cost (T146) -----

def fig_recovery_cost(out):
    """Cost of a hallucination as a function of when it is caught.

    T146 (Hallucination Consequence and Recovery): recovery cost is a function
    of cascade depth at detection. The undetected case is categorically
    different because it has no termination mechanism.
    """
    stages = [
        ("Caught before\nyou act on it", 1, GREEN, "costs a correction"),
        ("Caught in the\nnext reply", 3, GREEN, "costs an admission"),
        ("Caught a few\nsteps later", 9, AMBER, "costs rework"),
        ("Caught after the\nsession ended", 27, RED, "costs reconstruction"),
        ("Never caught", 90, "#7f1d1d", "compounds through everything\nbuilt on top of it"),
    ]

    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    x = np.arange(len(stages))
    vals = [s[1] for s in stages]
    ax.bar(x, vals, color=[s[2] for s in stages], width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=10.5)
    ax.set_ylabel("cost to recover", fontsize=10)
    ax.set_yticks([])
    ax.set_ylim(0, 118)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    for xi, (lbl, v, col, note) in zip(x, stages):
        ax.text(xi, v + 4, note, ha="center", va="bottom", fontsize=9,
                color=col if isinstance(col, str) else INK)

    ax.annotate("", xy=(4, 112), xytext=(4, 92),
                arrowprops=dict(arrowstyle="-|>", color="#7f1d1d", lw=2.4))
    ax.text(4, 114, "no ceiling", ha="center", fontsize=10,
            color="#7f1d1d", fontweight="bold")

    ax.set_title("What a hallucination costs depends almost entirely on when you catch it",
                 fontsize=13.5, fontweight="bold", pad=16, loc="left")
    fig.text(0.5, 0.005,
             "The last bar is not larger than the others. It is a different kind of thing: an undetected error has no "
             "termination point, so it keeps recruiting evidence and compounding through whatever gets built on top of it.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig5_recovery_cost.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------ fig 6: the consequence asymmetry -------

def fig_asymmetry(out):
    """Why the same failure recurs. T143 (SDCA) and T144 (SCB).

    The session boundary resets the model's behavioural learning. It does not
    reset the human's accumulated evidence. That asymmetry removes the
    consequence signal that would otherwise constrain the behaviour.
    """
    fig, ax = blank(12.4, 5.6)
    title(ax, "Why the same failure keeps coming back",
          "The session boundary resets one side of the relationship and not the other")

    n = 5
    w, gap = 14.4, 3.0
    x0 = 9          # leaves room for the row labels, which the box pad overruns

    # human row: memory carries across
    ax.text(2, 68, "YOU", fontsize=12, fontweight="bold", color=BLUE, va="center")
    for i in range(n):
        x = x0 + i * (w + gap)
        box(ax, x, 60, w, 14, f"session {i + 1}", fc=BLUE_L, ec=BLUE, fs=9)
    arrow(ax, (x0, 56), (x0 + n * (w + gap) - gap, 56), color=BLUE, lw=2.2)
    ax.text((x0 + n * (w + gap) - gap) / 2 + 2, 51,
            "your memory of what it did carries straight through",
            ha="center", fontsize=9.5, color=BLUE, style="italic")

    # model row: memory resets at each boundary
    ax.text(2, 34, "IT", fontsize=12, fontweight="bold", color=RED, va="center")
    for i in range(n):
        x = x0 + i * (w + gap)
        box(ax, x, 26, w, 14, f"session {i + 1}", fc=RED_L, ec=RED, fs=9)
        if i:
            ax.plot([x - gap / 2, x - gap / 2], [23, 43], color=RED, lw=2.2,
                    ls=(0, (2, 2)))
            ax.text(x - gap / 2, 21, "reset", ha="center", fontsize=8,
                    color=RED, style="italic")
    ax.text((x0 + n * (w + gap) - gap) / 2 + 2, 15,
            "it starts from zero every time, with no record that this happened before",
            ha="center", fontsize=9.5, color=RED, style="italic")

    box(ax, 6, 2, 88, 10,
        "It is not concealing the pattern from you. It cannot represent that there IS a pattern,\n"
        "so nothing it does is priced against what the last five sessions cost you.",
        fc=PAPER, ec=INK, fs=10.5)

    fig.tight_layout()
    fig.savefig(out / "fig6_asymmetry.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    figure_cli([fig_recovery_cost, fig_asymmetry],
               Path(__file__).resolve().parent / "figures",
               description=__doc__)
