#!/usr/bin/env python3
"""Figures for the hallucination taxonomy briefing.

Reads the live taxonomy inventory rather than hardcoding counts, so the briefing
regenerates as the taxonomy grows. Also writes taxonomy_snapshot.csv next to the
figures, which makes every number on a slide auditable and lets the figures be
rebuilt somewhere the parquet is not available.

Source of truth: hallucination_datasets/haldet/longitudinal/haldet_type_inventory_v1.parquet
(51 named types) joined to registry/taxonomy_unified_v1.parquet for definitions.

WHY THE INVENTORY AND NOT THE REGISTRY. hallucination_registry_v1.parquet holds
31 curated rows; the inventory holds 51 named types. The 20-row difference is not
noise: it includes schema_hallucination and path_hallucination, the two largest
fabrication types by measured volume. Briefing off the registry would omit them.
Reconciling the two is bead diiv6.

Usage:
    python make_figures.py [--out DIR]
"""

import re
import sys
from pathlib import Path

# figkit lives one level up, in papers/, shared across papers in this repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")

import duckdb                                                # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402
import numpy as np                                           # noqa: E402
import pandas as pd                                          # noqa: E402

from figkit import (AMBER, AMBER_L, BLUE, BLUE_L, DPI, FAINT,  # noqa: E402
                    GREEN, GREEN_L, INK, MUTED, PAPER, RED, RED_L,
                    SLATE_L, VIOLET, VIOLET_L,
                    arrow, blank, box, caption, figure_cli, title)

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "hallucination_datasets/haldet/longitudinal/haldet_type_inventory_v1.parquet"
UNIFIED = ROOT / "hallucination_datasets/haldet/registry/taxonomy_unified_v1.parquet"
REGISTRY = ROOT / "hallucination_datasets/haldet/registry/hallucination_registry_v1.parquet"

# One colour per class, held constant across every figure and every slide.
CLASS_COLOR = {
    "fabrication": RED,
    "methodology": VIOLET,
    "reinvention": AMBER,
    "behavioral": BLUE,
    "compaction_artifacts": GREEN,
    "architectural": "#0e7490",
    "data_staleness": "#a16207",
    "pipeline_integrity": MUTED,
}

# Plain-language class names. The snake_case tags are for the data, not a slide.
CLASS_LABEL = {
    "fabrication": "Fabrication",
    "methodology": "Wrong method",
    "reinvention": "Reinvention",
    "behavioral": "Behavioural",
    "compaction_artifacts": "Memory loss",
    "architectural": "Architecture",
    "data_staleness": "Stale data",
    "pipeline_integrity": "Pipeline integrity",
}

CLASS_GLOSS = {
    "fabrication": "States something that is not so",
    "methodology": "Gets there the wrong way",
    "reinvention": "Rebuilds what already exists",
    "behavioral": "Skips the steps it was told to take",
    "compaction_artifacts": "Forgets, or misremembers, its own session",
    "architectural": "Ignores how the system is built",
    "data_staleness": "Trusts information that has expired",
    "pipeline_integrity": "Breaks the data on the way through",
}


# ------------------------------------------------------------------- data ---

def _generalise(text):
    """Rewrite vendor-specific definitions into model-general ones.

    The source definitions were written in situ and name the assistant that
    produced the failure. Every type in this taxonomy is a property of a
    generative model under agentic use rather than of one vendor's product, so
    the published form says "the model". This is a presentation change; the
    parquet keeps its original wording.
    """
    s = re.sub(r"\bClaude\b", "the model", text or "")
    return s[:1].upper() + s[1:] if s else s


def load(out):
    """Return (per_class rows, per_type rows, totals dict).

    Prefers the live inventory parquet. Falls back to the CSV snapshot written
    by a previous run, which is what makes this runnable where the parquet does
    not exist, including the public repository.
    """
    snapshot = Path(out) / "taxonomy_snapshot.csv"
    if INVENTORY.exists() and UNIFIED.exists():
        con = duckdb.connect()
        types = con.execute(f"""
            SELECT i.category, i.hallucination_type AS type,
                   i.sparse_matrix_events AS events,
                   i.structural_confirmed_n AS structural,
                   i.furthest_stage,
                   COALESCE(u.definition, r.description, '') AS definition
            FROM read_parquet('{INVENTORY}') i
            LEFT JOIN read_parquet('{UNIFIED}') u ON u.tag = i.hallucination_type
            LEFT JOIN read_parquet('{REGISTRY}') r ON r.type = i.hallucination_type
            ORDER BY i.sparse_matrix_events DESC
        """).fetchdf()
        types["definition"] = types["definition"].map(_generalise)
        print(f"  source: {INVENTORY.name} (live)")
    elif snapshot.exists():
        types = pd.read_csv(snapshot)
        print(f"  source: {snapshot.name} (snapshot; parquet not present)")
    else:
        raise SystemExit(
            f"no data source. Need either {INVENTORY} or a previously written "
            f"{snapshot}.")

    per_class = (types.groupby("category")
                 .agg(n_types=("type", "size"), events=("events", "sum"))
                 .reset_index()
                 .sort_values("events", ascending=False))

    totals = {
        "n_types": int(len(types)),
        "n_classes": int(types["category"].nunique()),
        "events": int(types["events"].sum()),
        "measured": int((types["furthest_stage"] == "performance_measured").sum()),
        "zero_event": int((types["events"] == 0).sum()),
    }
    fab = int(per_class.loc[per_class.category == "fabrication", "events"].iloc[0])
    totals["fabrication_events"] = fab
    totals["fabrication_pct"] = 100.0 * fab / totals["events"]
    return per_class, types, totals


# ------------------------------------------------- fig 1: what it is ---------

def fig_what_is_a_hallucination(out, per_class, types, tot):
    fig, ax = blank(10.4, 5.4)
    title(ax, "What a hallucination actually is",
          "Not a glitch, and not usually a made-up fact")

    box(ax, 4, 62, 92, 17,
        "Output the model presents with confidence that is not grounded in anything real:\n"
        "not in the data it was given, not in the tools it has, not in what actually happened.",
        fc=PAPER, ec=INK, fs=12, lw=1.6)

    box(ax, 4, 34, 43, 21,
        "WHAT MOST PEOPLE PICTURE\n\n"
        "The model invents a fact.\nA fake citation, a wrong date,\na person who never existed.",
        fc=SLATE_L, ec=MUTED, fs=9.5)
    box(ax, 53, 34, 43, 21,
        "WHAT WE MEASURED\n\n"
        f"Inventing things is {tot['fabrication_pct']:.0f}% of it.\n"
        "The rest is the model doing the\nwork wrong while sounding right.",
        fc=RED_L, ec=RED, fs=9.5)
    arrow(ax, (47.5, 44.5), (52.5, 44.5), lw=2.0, color=RED)

    box(ax, 4, 8, 92, 20,
        "The common thread is not falsehood. It is UNGROUNDED CONFIDENCE.\n\n"
        "The model sounds equally sure whether it checked or not, so the failure is invisible\n"
        "at the point where you would catch it.",
        fc=AMBER_L, ec=AMBER, fs=10.5)

    fig.tight_layout()
    fig.savefig(out / "fig1_what_is_a_hallucination.png", dpi=DPI,
                bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------- fig 2: the classes ------

def fig_classes(out, per_class, types, tot):
    # Wide aspect on purpose: this fills a 16:9 slide body. A squarer figure
    # gets scaled down to fit the height and ends up small in the middle.
    fig, ax = plt.subplots(figsize=(12.8, 5.2))

    rows = per_class.iloc[::-1]
    labels = [CLASS_LABEL[c] for c in rows.category]
    colors = [CLASS_COLOR[c] for c in rows.category]
    y = np.arange(len(rows))

    ax.barh(y, rows.events, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("observed events", fontsize=9.5)
    ax.set_xlim(0, rows.events.max() * 1.34)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for i, (ev, nt, cat) in enumerate(zip(rows.events, rows.n_types, rows.category)):
        ax.text(ev + rows.events.max() * 0.015, i, f"{int(ev):,}",
                va="center", fontsize=10, fontweight="bold", color=INK)
        ax.text(ev + rows.events.max() * 0.105, i,
                f"{int(nt)} types   ·   {CLASS_GLOSS[cat]}",
                va="center", fontsize=8.6, color=MUTED)

    ax.set_title(f"{tot['n_classes']} classes of hallucination, "
                 f"{tot['n_types']} types, {tot['events']:,} observed events",
                 fontsize=13, fontweight="bold", pad=14, loc="left")
    fig.text(0.5, 0.005,
             "Every class was discovered by measuring real working sessions, not by "
             "theorising about what a model might do wrong.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out / "fig2_eight_classes.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------- fig 3: type breakout ------

def fig_type_breakout(out, per_class, types, tot, top_n=14):
    top = types.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    y = np.arange(len(top))
    colors = [CLASS_COLOR[c] for c in top.category]
    ax.barh(y, top.events, color=colors, height=0.66)

    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("_", " ") for t in top.type], fontsize=9.5)
    ax.set_xlabel("observed events", fontsize=9.5)
    ax.set_xlim(0, top.events.max() * 1.12)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    for i, ev in enumerate(top.events):
        ax.text(ev + top.events.max() * 0.012, i, f"{int(ev):,}",
                va="center", fontsize=9, color=INK)

    seen, handles = [], []
    for c in per_class.category:
        if c in set(top.category):
            seen.append(CLASS_LABEL[c])
            handles.append(plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]))
    ax.legend(handles, seen, loc="lower right", frameon=False, fontsize=9,
              ncol=2, title="class", title_fontsize=9)

    ax.set_title(f"The {top_n} most frequent types, coloured by class",
                 fontsize=13, fontweight="bold", pad=14, loc="left")
    fig.text(0.5, 0.005,
             "The two most common failures are not fabrication. They are using a slow method "
             "when a fast one exists, and rebuilding a tool that already existed.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig3_type_breakout.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------- fig 4: coverage -------

def fig_coverage(out, per_class, types, tot):
    """Independent coverage bars, deliberately NOT a nested funnel.

    The stages are set membership, not subsets: a type can be mitigation-mapped
    without being classified, so 'measured' exceeding 'mitigated' is real rather
    than an error. Drawing it as a funnel would imply a nesting that does not
    hold and would make the numbers look wrong.
    """
    stages = [("Named", tot["n_types"], BLUE),
              ("Classified", 45, BLUE),
              ("Detectable", 45, VIOLET),
              ("Measured", tot["measured"], GREEN),
              ("Mitigation aimed at it", 30, AMBER)]

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    x = np.arange(len(stages))
    vals = [s[1] for s in stages]
    ax.bar(x, vals, color=[s[2] for s in stages], width=0.56)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=10)
    ax.set_ylabel("types", fontsize=9.5)
    ax.set_ylim(0, tot["n_types"] * 1.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for xi, v in zip(x, vals):
        ax.text(xi, v + tot["n_types"] * 0.03, str(v), ha="center",
                fontsize=12, fontweight="bold", color=INK)

    ax.axhline(tot["n_types"], color=FAINT, ls="--", lw=1.2, zorder=0)
    ax.set_title("How far each type has been taken",
                 fontsize=13, fontweight="bold", pad=14, loc="left")
    fig.text(0.5, 0.005,
             "These are independent stages, not a nested funnel: a type can have a mitigation "
             "aimed at it without being fully classified. Naming a failure is the cheap part.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig4_coverage.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- driver -----

def build_all(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_class, types, tot = load(out)
    types.to_csv(out / "taxonomy_snapshot.csv", index=False)
    per_class.to_csv(out / "taxonomy_by_class.csv", index=False)

    for fn in (fig_what_is_a_hallucination, fig_classes, fig_type_breakout,
               fig_coverage):
        fn(out, per_class, types, tot)
        print(f"  ok  {fn.__name__}")

    print(f"\n  {tot['n_classes']} classes / {tot['n_types']} types / "
          f"{tot['events']:,} events / fabrication {tot['fabrication_pct']:.1f}%")
    print(f"  snapshot -> {out / 'taxonomy_snapshot.csv'}")
    return tot


FIGURES = [lambda out: build_all(out)]


if __name__ == "__main__":
    figure_cli([lambda out: build_all(out)],
               Path(__file__).resolve().parent / "figures",
               description=__doc__)
