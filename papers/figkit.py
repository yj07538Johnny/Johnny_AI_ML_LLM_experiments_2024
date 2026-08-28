#!/usr/bin/env python3
"""figkit — a small matplotlib toolkit for diagram-style paper and deck figures.

Extracted from the information-need cross-encoder figure set so the next paper
does not copy sixty lines of palette and helper boilerplate.

WHAT PROBLEM THIS SOLVES. Drawing block diagrams in matplotlib is unpleasant
because you are fighting data coordinates. `blank()` hands you an axes that is
0 to 100 in both directions with the frame switched off, so a box placed at
(4, 68, 20, 9) reads as "4% from the left, 68% up, 20% wide, 9% tall". You can
sketch a layout on paper and type it in.

USAGE

    from figkit import (blank, box, arrow, title, caption, render_all,
                        BLUE, BLUE_L, AMBER, AMBER_L, GREEN, INK, MUTED)

    def fig_example(out):
        fig, ax = blank(10.2, 5.0)
        title(ax, "Headline", "optional subtitle")
        box(ax, 10, 60, 30, 20, "a thing", fc=BLUE_L, ec=BLUE)
        arrow(ax, (40, 70), (55, 70))
        save(fig, out, "fig01_example")

    if __name__ == "__main__":
        render_all([fig_example])

COLOUR DISCIPLINE. The palette below is the single source of truth for both the
figures and the slide deck, so a colour means the same thing in both. Assign a
role to each hue once and keep it: in the cross-encoder set, blue is always the
report side and amber always the need side. That single rule is most of why a
figure set reads as one system.

Deck code needs pptx RGBColor rather than hex strings, so use `rgb()`:

    from pptx.dml.color import RGBColor
    from figkit import rgb, BLUE
    PPTX_BLUE = RGBColor(*rgb(BLUE))

figkit itself imports only matplotlib, never pptx.

VERIFY BY LOOKING. Layout collisions are invisible in source and obvious in the
rendered PNG. Render, open the image, fix the overlap, repeat. Every layout bug
in the original set was found that way and none were found by reading code.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.patches import FancyBboxPatch                # noqa: E402

# ---------------------------------------------------------------- palette ---

INK = "#111827"        # body text
MUTED = "#6b7280"      # secondary text, default strokes
FAINT = "#d1d5db"      # dividers
PAPER = "#ffffff"      # background
SLATE_L = "#f1f5f9"    # neutral fill

BLUE = "#2563eb"
BLUE_L = "#dbeafe"
AMBER = "#b45309"
AMBER_L = "#fef3c7"
GREEN = "#047857"
GREEN_L = "#d1fae5"
RED = "#b91c1c"
RED_L = "#fee2e2"
VIOLET = "#6d28d9"
VIOLET_L = "#ede9fe"

# --- deck variants -----------------------------------------------------------
# Slides need two things figures do not. Fills sit behind body text, so the _L
# tints above are too saturated; and section dividers use a full-bleed dark
# ground with light text on top. Same hues, different jobs.

INK_DARK = "#0f172a"      # slide title bars and divider grounds
GREEN_S = "#ecfdf5"       # callout fills, lighter than the _L tints
RED_S = "#fef2f2"
VIOLET_S = "#ede9fe"

BLUE_ON_DARK = "#93c5fd"  # text tints for use on a dark ground
GREEN_ON_DARK = "#a7f3d0"
AMBER_ON_DARK = "#fde68a"
VIOLET_ON_DARK = "#ddd6fe"
RED_ON_DARK = "#fecaca"

AMBER_DEEP = "#92400e"    # divider grounds, one per talk section
VIOLET_DEEP = "#5b21b6"
RED_DEEP = "#991b1b"
GREEN_DEEP = "#065f46"

#: name -> hex, for callers that would rather look colours up than import them.
HEX = {
    "ink": INK, "muted": MUTED, "faint": FAINT, "paper": PAPER,
    "slate_l": SLATE_L,
    "blue": BLUE, "blue_l": BLUE_L, "amber": AMBER, "amber_l": AMBER_L,
    "green": GREEN, "green_l": GREEN_L, "red": RED, "red_l": RED_L,
    "violet": VIOLET, "violet_l": VIOLET_L,
}

DPI = 200
FONT = "DejaVu Sans"


def rgb(hexstr: str) -> tuple[int, int, int]:
    """'#2563eb' -> (37, 99, 235). For pptx RGBColor, which wants ints."""
    h = hexstr.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"expected a 6-digit hex colour, got {hexstr!r}")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ------------------------------------------------------------------ style ---

def use_style(font: str = FONT, size: float = 9) -> None:
    """Apply the shared rcParams. Called automatically on import."""
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "font.family": font,
        "font.size": size,
        "text.color": INK,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


use_style()


# ------------------------------------------------------------- primitives ---

def blank(w: float, h: float):
    """A figure sized (w, h) inches with one axes on a 0-100 percentage grid."""
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, fc=SLATE_L, ec=None, fs=9, tc=INK,
        lw=1.3, weight="normal", pad=0.4, z=2):
    """A rounded rectangle with centred, possibly multi-line, text.

    x, y is the lower-left corner in percentage coordinates. Pass fc/ec as a
    light/dark pair from the palette (BLUE_L with BLUE) for a filled box that
    still reads at a distance.
    """
    ec = ec if ec is not None else MUTED
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad={pad}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=z + 1, linespacing=1.4,
            fontweight=weight)


def arrow(ax, p0, p1, color=MUTED, lw=1.5, rad=0.0, ls="-", style="-|>", z=3):
    """An arrow from p0 to p1, both (x, y) percentage tuples.

    `rad` bends it: positive curves one way, negative the other. Use a small
    bend (0.1 to 0.2) to route a connector around a box instead of through it.
    Pass style="-" for a plain line segment with no head.
    """
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw,
                                linestyle=ls, shrinkA=1, shrinkB=1,
                                connectionstyle=f"arc3,rad={rad}"))


def title(ax, text, sub=None):
    """Figure headline at the top of a blank() axes, with optional subtitle."""
    ax.text(50, 96, text, ha="center", va="top", fontsize=12.5,
            fontweight="bold", color=INK)
    if sub:
        ax.text(50, 89.5, sub, ha="center", va="top", fontsize=8.8, color=MUTED)


def caption(ax, text, y=2.5):
    """A muted italic note along the bottom of a blank() axes."""
    ax.text(50, y, text, ha="center", va="bottom", fontsize=8, color=MUTED,
            style="italic")


def save(fig, out_dir, name, dpi=DPI, close=True):
    """Write fig to out_dir/name.png and close it. Returns the path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    if close:
        plt.close(fig)
    return path


# ------------------------------------------------------------------ driver --

def render_all(figures, out, verbose=True):
    """Call every figure function with `out`, then report what landed.

    Each function takes the output directory and writes its own PNG, which keeps
    naming next to the drawing code instead of in a lookup table.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    for fn in figures:
        fn(out)
        if verbose:
            print(f"  ok  {fn.__name__}")

    made = sorted(out.glob("*.png"))
    if verbose:
        print(f"\n{len(made)} figures -> {out}")
        for p in made:
            print(f"  {p.name:44s} {p.stat().st_size / 1024:7.1f} KB")
    return made


def figure_cli(figures, default_out, description=None):
    """Standard `--out DIR` entry point. Wire it up as:

        if __name__ == "__main__":
            figure_cli(FIGURES, Path(__file__).resolve().parent / "figures")
    """
    import argparse

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out", default=str(default_out))
    args = ap.parse_args()
    return render_all(figures, args.out)
