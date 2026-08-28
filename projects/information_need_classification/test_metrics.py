#!/usr/bin/env python3
"""Checks for the metric and data layers, against hand-computable cases.

These matter because the reported numbers are the deliverable. A metric that is
silently wrong produces a confident wrong conclusion, which is worse than a
crash. No torch, so this runs anywhere.

    python test_metrics.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import metrics
from data import SamplingPolicy, build_pairs, split_by_text

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}   {detail}")
        FAILS.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) < tol


# ------------------------------------------------------------ recall@k ------

def test_recall_at_k():
    # 2 texts, 4 needs. Text 0 gold = {n0}, ranked 1st by score. Text 1 gold =
    # {n3}, ranked last. So recall@1 = (1 + 0)/2 = 0.5, recall@4 = 1.0.
    S = np.array([[0.9, 0.8, 0.7, 0.6],
                  [0.9, 0.8, 0.7, 0.6]])
    Y = np.array([[1, 0, 0, 0],
                  [0, 0, 0, 1]])
    out = metrics.recall_at_k(S, Y, ks=(1, 2, 4))
    check("recall@1 = 0.5", approx(out["recall@1"], 0.5), out)
    check("recall@4 = 1.0", approx(out["recall@4"], 1.0), out)

    # A text with NO gold needs must be excluded, not counted as a hit. If it
    # were counted the score would drift toward 1 for free.
    S3 = np.vstack([S, [[0.9, 0.8, 0.7, 0.6]]])
    Y3 = np.vstack([Y, [[0, 0, 0, 0]]])
    out3 = metrics.recall_at_k(S3, Y3, ks=(1,))
    check("texts with no gold are excluded from recall",
          approx(out3["recall@1"], 0.5) and out3["n_texts_with_gold"] == 2, out3)

    # k larger than the candidate set is skipped, not silently clamped.
    check("k > n_needs is omitted", "recall@99" not in
          metrics.recall_at_k(S, Y, ks=(1, 99)))


# ----------------------------------------------------------------- mAP ------

def test_map():
    # Perfect ranking: the single gold need scores highest. AP = 1.0 both rows.
    S = np.array([[0.9, 0.1, 0.2], [0.1, 0.95, 0.3]])
    Y = np.array([[1, 0, 0], [0, 1, 0]])
    check("mAP = 1.0 on a perfect ranking",
          approx(metrics.mean_average_precision(S, Y), 1.0))

    # Gold ranked last of three -> AP = 1/3.
    S2 = np.array([[0.1, 0.9, 0.8]])
    Y2 = np.array([[1, 0, 0]])
    check("mAP = 1/3 when the gold need ranks last",
          approx(metrics.mean_average_precision(S2, Y2), 1 / 3))

    # Rows with no positives contribute nothing rather than NaN-poisoning.
    S3 = np.vstack([S, [[0.5, 0.5, 0.5]]])
    Y3 = np.vstack([Y, [[0, 0, 0]]])
    check("all-negative rows are skipped, not NaN",
          approx(metrics.mean_average_precision(S3, Y3), 1.0))


# --------------------------------------------------------- micro / macro ----

def test_micro_macro_gap():
    """The gap must widen when a rare need fails and a frequent one succeeds.

    This is the headline diagnostic, so it gets a directional test rather than
    a point value.
    """
    need_ids = ["frequent", "rare"]
    recs_good, recs_bad = [], []
    for i in range(20):                       # 20 texts, all gold on 'frequent'
        recs_good.append((f"t{i}", "frequent", 0.9, 1))
        recs_bad.append((f"t{i}", "frequent", 0.9, 1))
        recs_good.append((f"t{i}", "rare", 0.1, 0))
        recs_bad.append((f"t{i}", "rare", 0.1, 0))
    # two texts also carry the rare need; in the bad run the model misses both
    for i in (0, 1):
        recs_good = [r for r in recs_good if not (r[0] == f"t{i}" and r[1] == "rare")]
        recs_bad = [r for r in recs_bad if not (r[0] == f"t{i}" and r[1] == "rare")]
        recs_good.append((f"t{i}", "rare", 0.95, 1))     # found it
        recs_bad.append((f"t{i}", "rare", 0.05, 1))      # missed it

    good = metrics.evaluate(recs_good, need_ids, threshold=0.5)
    bad = metrics.evaluate(recs_bad, need_ids, threshold=0.5)
    check("gap is ~0 when the rare need is handled",
          abs(good["micro_macro_gap"]) < 0.05, good["micro_macro_gap"])
    check("gap widens when the rare need fails",
          bad["micro_macro_gap"] > good["micro_macro_gap"] + 0.2,
          f"good={good['micro_macro_gap']:.3f} bad={bad['micro_macro_gap']:.3f}")
    check("micro-F1 barely moves while macro-F1 collapses",
          bad["micro_f1"] > 0.8 and bad["macro_f1_needs_with_support"] < 0.6,
          f"micro={bad['micro_f1']:.3f} macro={bad['macro_f1_needs_with_support']:.3f}")


# ------------------------------------------------------------ thresholds ----

def test_gold_map_counts_unscored_needs_as_misses():
    """A gold need the retriever never surfaced must count against recall.

    Without gold_map the truth is read off the scored records, so a need that
    produced no record is invisible rather than wrong, and recall is computed
    over the shortlist instead of over the truth. That would let the reported
    number exceed the retriever's own ceiling, which the served system cannot do.
    """
    need_ids = ["n0", "n1", "n2", "n3"]
    # The system scored only n0 and n1. Gold is {n0, n3}; n3 was never surfaced.
    records = [("t0", "n0", 0.99, 1), ("t0", "n1", 0.10, 0)]
    gold_map = {"t0": ["n0", "n3"]}

    naive = metrics.evaluate(records, need_ids, threshold=0.5, ks=(1, 3))
    honest = metrics.evaluate(records, need_ids, threshold=0.5,
                              gold_map=gold_map, ks=(1, 3, 4))

    check("without gold_map the missed need is invisible (recall@1 = 1.0)",
          approx(naive["recall@1"], 1.0), naive["recall@1"])
    check("with gold_map the missed need counts (recall@1 = 0.5)",
          approx(honest["recall@1"], 0.5), honest["recall@1"])
    # An unscored need never competes for a top-k slot, so it is unreachable at
    # EVERY k, including k equal to the whole vocabulary. That is the property
    # two-stage serving needs: a need the retriever never surfaced cannot be
    # recovered by widening k after the fact, and the metric must say so.
    check("an unscored gold need is missed at k < |vocabulary|",
          approx(honest["recall@3"], 0.5), honest["recall@3"])
    check("an unscored gold need is missed even at k = |vocabulary|",
          approx(honest["recall@4"], 0.5), honest["recall@4"])
    check("gold_map lowers recall, never raises it",
          honest["recall@1"] <= naive["recall@1"])
    check("micro-recall also drops, since the miss is a false negative",
          honest["micro_recall"] < naive["micro_recall"],
          f"{honest['micro_recall']} vs {naive['micro_recall']}")


def test_thresholds():
    # Scores separate cleanly at 0.5, so the tuned threshold must land between
    # the two clusters and reach F1 = 1.0.
    S = np.array([[0.9, 0.1], [0.85, 0.2], [0.15, 0.95]])
    Y = np.array([[1, 0], [1, 0], [0, 1]])
    t, f1 = metrics.tune_global_threshold(S, Y)
    check("global threshold separates the clusters", 0.2 < t <= 0.85, t)
    check("global threshold reaches F1 = 1.0", approx(f1, 1.0, 1e-6), f1)

    # Per-need fitting must respect min_support: with support 2 and a threshold
    # of 10, both needs fall back to the global value.
    per_need, n_fitted = metrics.tune_per_need_thresholds(
        S, Y, ["a", "b"], t, min_support=10)
    check("thin needs fall back to the global threshold",
          n_fitted == 0 and set(per_need.values()) == {t}, per_need)

    per_need2, n_fitted2 = metrics.tune_per_need_thresholds(
        S, Y, ["a", "b"], t, min_support=1)
    check("needs with support are fitted individually", n_fitted2 == 2, n_fitted2)


# ------------------------------------------------------------------ data ----

def test_split_by_text():
    texts = pd.DataFrame({
        "text_id": [f"d{i}" for i in range(100)],
        "text": ["x"] * 100,
        "need_ids": [["n0"]] * 100,
    })
    out = split_by_text(texts, val_frac=0.2, test_frac=0.2, seed=7)
    counts = out["split"].value_counts().to_dict()
    check("split sizes are as requested",
          counts.get("val") == 20 and counts.get("test") == 20, counts)
    check("every text lands in exactly one split",
          len(out) == 100 and out["text_id"].nunique() == 100)
    again = split_by_text(texts, val_frac=0.2, test_frac=0.2, seed=7)
    check("split is deterministic for a fixed seed",
          out["split"].tolist() == again["split"].tolist())


def test_negative_sampling_reports_shortfall():
    texts = pd.DataFrame({
        "text_id": ["d0"], "text": ["x"], "need_ids": [["n0"]],
    })
    needs = pd.DataFrame({
        "need_id": [f"n{i}" for i in range(10)],
        "need_label": ["l"] * 10,
        "justification": ["j"] * 10,
    })
    policy = SamplingPolicy(n_hard=4, n_random=4, seed=1)

    # No retriever supplied: the hard quota cannot be met and must be declared.
    _, stats = build_pairs(texts, needs, policy, hard_negatives=None)
    check("hard-negative shortfall is reported, not absorbed",
          stats["hard_satisfied"] == 0 and "WARNING" in stats, stats)
    check("shortfall is backfilled so the ratio still holds",
          stats["n_negative"] == 8, stats)

    # With a retriever, gold must be filtered out of the hard candidates.
    ranked = {"d0": ["n0", "n1", "n2", "n3", "n4"]}
    pairs, stats2 = build_pairs(texts, needs, policy, hard_negatives=ranked)
    hard_ids = {p.need_id for p in pairs if p.source == "hard"}
    check("gold needs never become hard negatives", "n0" not in hard_ids, hard_ids)
    check("hard quota satisfied when candidates exist",
          stats2["hard_satisfied"] == 4, stats2)
    check("no pair is both positive and negative",
          not ({p.need_id for p in pairs if p.label == 1} & hard_ids))


def main():
    print("\n  metrics")
    test_recall_at_k()
    test_map()
    test_micro_macro_gap()
    test_gold_map_counts_unscored_needs_as_misses()
    test_thresholds()
    print("\n  data")
    test_split_by_text()
    test_negative_sampling_reports_shortfall()

    print()
    if FAILS:
        print(f"  {len(FAILS)} FAILED: {FAILS}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
