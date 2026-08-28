#!/usr/bin/env python3
"""Figures for the compaction hallucination briefing.

Reads the live audit and event parquets, and writes CSV snapshots beside the
figures so every number on a slide is auditable and the figures rebuild where
the parquets are not available.

Sources
  hallucination_datasets/haldet/output/compaction_audit_v1.parquet      337 audited summaries
  hallucination_datasets/haldet/longitudinal/compaction_events_v1.parquet  signal timeline
  ~/.claude/session-memory-datalake/raw_messages/*.parquet              session denominators

WHY THE DENOMINATOR MATTERS HERE. Monthly session volume swings by more than 20x
across this window (97 to 2,217). Raw compaction counts across that boundary
produce a clean "it collapsed after the intervention" story that normalising
does not fully support. Figure 4 draws both on purpose. Reporting only the raw
panel would be the same error this taxonomy exists to catch.

Usage:
    python make_figures.py [--out DIR]
"""

import sys
from pathlib import Path

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

import os                                                    # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "hallucination_datasets/haldet/output/compaction_audit_v1.parquet"
EVENTS = ROOT / "hallucination_datasets/haldet/longitudinal/compaction_events_v1.parquet"

# Session-message parquets, used only for the per-month session denominators in
# figure 4. Machine-specific, so it is overridable and the run falls back to the
# CSV snapshots when it is absent.
RAW = Path(os.environ.get(
    "SESSION_DATALAKE",
    str(Path.home() / ".claude" / "session-memory-datalake"))) / "raw_messages" / "*.parquet"

INTERVENTION = "2026-02"     # the month the human practice began

# The eight audited dimensions, ordered as they are reported.
DIMS = [
    ("score_git_commit_accuracy",     "Git commit hashes",    "literal"),
    ("score_timeline_fidelity",       "Order of events",      "literal"),
    ("score_tool_coverage",           "Which tools were used", "mixed"),
    ("score_message_completeness",    "How many messages",    "mixed"),
    ("score_file_creation_accuracy",  "Which files changed",  "mixed"),
    ("score_message_content_accuracy", "What was actually said", "judgement"),
    ("score_error_identification",    "What went wrong",      "judgement"),
]
KIND_COLOR = {"literal": GREEN, "mixed": AMBER, "judgement": RED}


# ------------------------------------------------------------------- data ---

def load(out):
    """Live parquets if present, else the CSV snapshots written by a prior run."""
    out = Path(out)
    snap_dims = out / "fidelity_dimensions.csv"
    snap_month = out / "monthly_compaction.csv"
    snap_tot = out / "compaction_totals.csv"

    if AUDIT.exists() and EVENTS.exists():
        con = duckdb.connect()
        agg = con.execute(f"""
            SELECT COUNT(*) n_summaries, COUNT(DISTINCT session_id) n_sessions,
                   AVG(weighted_fidelity_score) mean_fidelity,
                   AVG(score_fabrication_rate) fabrication,
                   SUM(messages_actual) msgs_actual, SUM(messages_claimed) msgs_claimed,
                   SUM(errors_actual) err_actual,    SUM(errors_claimed) err_claimed,
                   SUM(files_actual) files_actual,   SUM(files_claimed) files_claimed,
                   SUM(tools_actual) tools_actual,   SUM(tools_claimed) tools_claimed,
                   SUM(CASE WHEN tags_detected = '[]' THEN 1 ELSE 0 END) n_untagged,
                   SUM(CASE WHEN errors_claimed < errors_actual THEN 1 ELSE 0 END) n_understate,
                   SUM(CASE WHEN errors_claimed > errors_actual THEN 1 ELSE 0 END) n_overstate
            FROM read_parquet('{AUDIT}')
        """).fetchdf().iloc[0].to_dict()

        dims = con.execute(f"""
            SELECT {', '.join(f'AVG({c}) AS "{c}"' for c, _, _ in DIMS)}
            FROM read_parquet('{AUDIT}')
        """).fetchdf().iloc[0]
        dims = pd.DataFrame([{"dimension": lbl, "kind": kind, "score": float(dims[c])}
                             for c, lbl, kind in DIMS])

        month = con.execute(f"""
            WITH ev AS (
              SELECT strftime(CAST(timestamp AS DATE), '%Y-%m') mon,
                     SUM(CASE WHEN signal_type='post_compaction_continuation' THEN 1 ELSE 0 END) compactions,
                     SUM(CASE WHEN signal_type='land_the_plane' THEN 1 ELSE 0 END) land_the_plane
              FROM read_parquet('{EVENTS}') GROUP BY 1),
            se AS (
              SELECT strftime(CAST(timestamp AS DATE), '%Y-%m') mon,
                     COUNT(DISTINCT session_id) sessions
              FROM read_parquet('{RAW}')
              WHERE timestamp >= '2025-12-01' AND timestamp < '2026-07-01' GROUP BY 1)
            SELECT ev.mon, ev.compactions, ev.land_the_plane, se.sessions
            FROM ev JOIN se USING (mon) ORDER BY ev.mon
        """).fetchdf()
        month["per_100_sessions"] = 100.0 * month.compactions / month.sessions

        out.mkdir(parents=True, exist_ok=True)
        dims.to_csv(snap_dims, index=False)
        month.to_csv(snap_month, index=False)
        pd.DataFrame([agg]).to_csv(snap_tot, index=False)
        print(f"  source: live parquets ({AUDIT.name}, {EVENTS.name})")
    elif snap_dims.exists() and snap_month.exists() and snap_tot.exists():
        dims = pd.read_csv(snap_dims)
        month = pd.read_csv(snap_month)
        agg = pd.read_csv(snap_tot).iloc[0].to_dict()
        print("  source: CSV snapshots (parquets not present)")
    else:
        raise SystemExit(f"no data source; need {AUDIT} or snapshots in {out}")

    agg = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
           for k, v in agg.items()}
    return dims, month, agg


# ------------------------------------------------- fig 1: what it is ---------

def fig_what_is_compaction(out, dims, month, agg):
    # Wide aspect to fill a 16:9 slide body rather than being letterboxed.
    fig, ax = blank(12.4, 5.2)
    title(ax, "What compaction is",
          "The model runs out of room, so it summarises its own history and works from the summary")

    box(ax, 2, 66, 20, 20, "A long session\n\nhours of work,\nthousands of steps",
        fc=BLUE_L, ec=BLUE, fs=9)
    box(ax, 26, 66, 21, 20, "Memory fills up\n\nthe context window\nhas a hard limit",
        fc=AMBER_L, ec=AMBER, fs=9)
    box(ax, 51, 66, 22, 20,
        "It summarises\nITSELF\n\nand the original\nrecord is gone", fc=RED_L, ec=RED, fs=9)
    box(ax, 77, 66, 21, 20,
        "Work continues\n\nfrom the summary,\nnot from what\nhappened", fc=SLATE_L, fs=9)
    for x0, x1 in ((22.5, 25.5), (47.5, 50.5), (73.5, 76.5)):
        arrow(ax, (x0, 76), (x1, 76), lw=1.8)

    box(ax, 4, 34, 92, 24,
        "This is not the model forgetting a fact it was told.\n"
        "It is the model rewriting the record of its own work, then trusting the rewrite.\n\n"
        f"Across {int(agg['n_summaries'])} audited summaries from "
        f"{int(agg['n_sessions'])} sessions, the average summary preserved\n"
        f"{agg['mean_fidelity'] * 100:.0f}% of what the session actually contained.",
        fc=PAPER, ec=INK, fs=11, lw=1.6)

    box(ax, 4, 8, 92, 20,
        "The part that surprises people: the model does not INVENT during this step.\n\n"
        "Measured fabrication rate across every audited summary: 0%.\n"
        "It loses things and it reshapes things. It does not make them up.",
        fc=GREEN_L, ec=GREEN, fs=10.5)

    fig.tight_layout()
    fig.savefig(out / "fig1_what_is_compaction.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------- fig 2: the fidelity gradient ----

def fig_fidelity_gradient(out, dims, month, agg):
    d = dims.iloc[::-1]
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    y = np.arange(len(d))
    ax.barh(y, d.score * 100, color=[KIND_COLOR[k] for k in d.kind], height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(d.dimension, fontsize=11)
    ax.set_xlabel("percent of the truth preserved in the summary", fontsize=9.5)
    ax.set_xlim(0, 118)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for i, v in enumerate(d.score * 100):
        # One decimal near the ceiling: 99.7 rounded to "100%" would claim a
        # perfection the data does not show.
        lbl = f"{v:.1f}%" if v >= 99 else f"{v:.0f}%"
        ax.text(v + 1.5, i, lbl, va="center", fontsize=11,
                fontweight="bold", color=INK)

    ax.axvline(agg["mean_fidelity"] * 100, color=MUTED, ls="--", lw=1.4, zorder=0)
    ax.text(agg["mean_fidelity"] * 100 + 1.5, len(d) - 0.4,
            f"overall {agg['mean_fidelity'] * 100:.0f}%", fontsize=9, color=MUTED)

    handles = [plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR[k])
               for k in ("literal", "mixed", "judgement")]
    ax.legend(handles,
              ["A literal token: it matches or it does not",
               "Partly literal, partly judgement",
               "Requires judgement about what happened"],
              loc="lower right", frameon=False, fontsize=9)

    ax.set_title("What survives the summary, and what does not",
                 fontsize=13.5, fontweight="bold", pad=14, loc="left")
    fig.text(0.5, 0.005,
             "The gradient is the finding. A commit hash is a string that either matches or does not, "
             "and it survives 997 times in 1,000. What went wrong in the session is a judgement, and it survives a third of the time.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out / "fig2_fidelity_gradient.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ fig 3: what gets dropped ---

def fig_what_gets_dropped(out, dims, month, agg):
    pairs = [("Tool calls", agg["tools_actual"], agg["tools_claimed"]),
             ("User messages", agg["msgs_actual"], agg["msgs_claimed"]),
             ("Files changed", agg["files_actual"], agg["files_claimed"]),
             ("Errors hit", agg["err_actual"], agg["err_claimed"])]

    fig, ax = plt.subplots(figsize=(12.2, 5.2))
    y = np.arange(len(pairs))
    h = 0.36
    ax.barh(y + h / 2, [p[1] for p in pairs], height=h, color=BLUE,
            label="actually happened")
    ax.barh(y - h / 2, [p[2] for p in pairs], height=h, color=FAINT,
            label="survived into the summary")
    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in pairs], fontsize=11.5)
    ax.set_xlabel("count across all 337 audited summaries", fontsize=9.5)
    ax.set_xlim(0, max(p[1] for p in pairs) * 1.26)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")

    for i, (_, a, c) in enumerate(pairs):
        lost = 100.0 * (a - c) / a
        ax.text(a * 1.02, i, f"{int(a):,} → {int(c):,}", va="center",
                fontsize=10, color=INK)
        ax.text(a * 1.02, i - 0.26, f"{lost:.0f}% lost", va="center",
                fontsize=9.5, color=RED, fontweight="bold")

    ax.set_title("How much of the session does not make it through",
                 fontsize=13.5, fontweight="bold", pad=14, loc="left")
    fig.text(0.5, 0.005,
             f"Of {int(agg['n_summaries'])} summaries, {int(agg['n_understate'])} under-report how many errors "
             f"occurred and {int(agg['n_overstate'])} over-report. The dominant direction is losing bad news, not inventing it.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(out / "fig3_what_gets_dropped.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------- fig 4: intervention -----

def fig_intervention(out, dims, month, agg):
    """Both readings, side by side, on purpose.

    Raw counts show a collapse. Normalised by session volume, which swings more
    than 20x across this window, the drop is real but partly reverts. Showing
    only the left panel would be the error this taxonomy exists to catch.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    x = np.arange(len(month))
    iv = list(month.mon).index(INTERVENTION) if INTERVENTION in list(month.mon) else None

    ax = axes[0]
    ax.bar(x, month.compactions, color=RED, width=0.6)
    ax.set_xticks(x), ax.set_xticklabels(month.mon, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("compaction events", fontsize=9.5)
    ax.set_title("Raw counts\nlooks like a clean collapse", fontsize=11.5,
                 fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for xi, v in zip(x, month.compactions):
        ax.text(xi, v + 1.5, str(int(v)), ha="center", fontsize=9.5, color=INK)

    ax = axes[1]
    ax.bar(x, month.per_100_sessions, color=VIOLET, width=0.6)
    ax.set_xticks(x), ax.set_xticklabels(month.mon, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("compactions per 100 sessions", fontsize=9.5)
    ax.set_title("Per 100 sessions\ndrops, then partly returns", fontsize=11.5,
                 fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for xi, v in zip(x, month.per_100_sessions):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9.5, color=INK)

    for ax in axes:
        if iv is not None:
            ax.axvline(iv - 0.5, color=GREEN, ls="--", lw=1.8)
            ax.text(iv - 0.62, ax.get_ylim()[1] * 0.97,
                    "human practice\nbegins", fontsize=8.6, color=GREEN,
                    va="top", ha="right")

    fig.suptitle("Did anything reduce it?", fontsize=14, fontweight="bold",
                 x=0.02, ha="left", y=0.99)
    fig.text(0.5, 0.005,
             "Same data, two denominators. Monthly session volume ranges from 97 to 2,217 across this window, "
             "so raw counts are not comparable month to month. The supportable reading is that it dropped and the effect is not yet established.",
             ha="center", fontsize=8.5, color=MUTED, style="italic")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out / "fig4_intervention.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------- driver -----

def build_all(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dims, month, agg = load(out)

    for fn in (fig_what_is_compaction, fig_fidelity_gradient,
               fig_what_gets_dropped, fig_intervention):
        fn(out, dims, month, agg)
        print(f"  ok  {fn.__name__}")

    print(f"\n  {int(agg['n_summaries'])} summaries / {int(agg['n_sessions'])} sessions / "
          f"mean fidelity {agg['mean_fidelity']:.4f} / fabrication {agg['fabrication']:.1f}")
    print(f"  untagged summaries: {int(agg['n_untagged'])} of {int(agg['n_summaries'])}")
    return agg


if __name__ == "__main__":
    figure_cli([lambda out: build_all(out)],
               Path(__file__).resolve().parent / "figures",
               description=__doc__)
