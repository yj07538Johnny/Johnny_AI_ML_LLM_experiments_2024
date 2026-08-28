#!/usr/bin/env python3
"""Multi-label metrics for information-need classification.

Implements Section 8 of the white paper. The paper argues two numbers matter
more than the headline F1:

  1. the GAP between micro-F1 and macro-F1  (does the long tail work?)
  2. recall@k from the retrieval stage      (what is the ceiling?)

so both are first-class here rather than derived by the caller.

Pure numpy + sklearn. No torch, so this is testable on its own.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support


# --------------------------------------------------------------- helpers ----

def _as_matrix(records, need_ids, gold_map=None):
    """records: iterable of (text_id, need_id, score, label).

    Returns (text_ids, need_ids, score_matrix, label_matrix), both matrices
    shaped (n_texts, n_needs) with NaN in S where a pair was not scored.

    gold_map ({text_id: iterable of need_id}) is how end-to-end evaluation stays
    honest. Under a two-stage system the cross-encoder only scores the
    retriever's shortlist, so a gold need the retriever missed produces NO
    record at all. Building Y from the records alone would make that need
    invisible instead of wrong, and recall would be computed over the shortlist
    rather than over the truth. It would then exceed the retriever's own ceiling,
    which is impossible for the system actually being served. Pass gold_map and
    the missed need appears in Y with a NaN score, ranks last, and counts as the
    miss it is.
    """
    need_ix = {n: j for j, n in enumerate(need_ids)}
    text_ids = sorted({r[0] for r in records} | set(gold_map or {}))
    text_ix = {t: i for i, t in enumerate(text_ids)}

    S = np.full((len(text_ids), len(need_ids)), np.nan, dtype=float)
    Y = np.zeros((len(text_ids), len(need_ids)), dtype=int)
    for tid, nid, score, label in records:
        if nid not in need_ix:
            continue
        i, j = text_ix[tid], need_ix[nid]
        S[i, j] = score
        if gold_map is None:
            Y[i, j] = int(label)

    if gold_map is not None:
        for tid, gold in gold_map.items():
            for nid in gold:
                if nid in need_ix:
                    Y[text_ix[tid], need_ix[nid]] = 1
    return text_ids, list(need_ids), S, Y


# ------------------------------------------------------------ thresholds ----

def tune_global_threshold(S, Y, grid=None):
    """Pick the single threshold maximising micro-F1. Validation only."""
    grid = grid if grid is not None else np.linspace(0.02, 0.98, 49)
    filled = np.nan_to_num(S, nan=-1.0)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred = (filled >= t).astype(int)
        f1 = f1_score(Y.ravel(), pred.ravel(), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def tune_per_need_thresholds(S, Y, need_ids, global_t, min_support=10, grid=None):
    """A threshold per need, falling back to the global one when support is thin.

    Section 8.1. Needs differ in base rate, so one cut point serves the common
    ones well and the rare ones badly. But a per-need threshold fitted on three
    validation examples is noise, hence min_support.
    """
    grid = grid if grid is not None else np.linspace(0.02, 0.98, 49)
    filled = np.nan_to_num(S, nan=-1.0)
    out, n_fitted = {}, 0
    for j, nid in enumerate(need_ids):
        support = int(Y[:, j].sum())
        if support < min_support:
            out[nid] = global_t
            continue
        best_t, best_f1 = global_t, -1.0
        for t in grid:
            f1 = f1_score(Y[:, j], (filled[:, j] >= t).astype(int),
                          zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = float(t), float(f1)
        out[nid] = best_t
        n_fitted += 1
    return out, n_fitted


# --------------------------------------------------------------- metrics ----

def recall_at_k(S, Y, ks=(1, 5, 10, 20, 50)):
    """Fraction of gold needs appearing in the top-k SCORED needs per text.

    Two exclusions, both of which would otherwise inflate the number:

    Texts with no gold needs are dropped. They cannot contribute recall, and
    counting them would drag the mean toward 1 for free.

    Needs that were never scored are unreachable at EVERY k, not just at small
    k. Ranking them by a filled-in sentinel would let them drift into the top-k
    in index order once k grew past the shortlist size, so a run that scored 10
    candidates would report a rising recall@50. Nothing was retrieved to make
    that true. Only finite scores compete for a top-k slot here.
    """
    out = {}
    has_gold = Y.sum(axis=1) > 0
    n_needs = S.shape[1]
    for k in ks:
        if k > n_needs:
            continue
        hits = []
        for i in np.where(has_gold)[0]:
            scored = np.where(np.isfinite(S[i]))[0]
            if scored.size:
                ranked = scored[np.argsort(-S[i, scored])][:k]
            else:
                ranked = np.array([], dtype=int)
            gold = set(np.where(Y[i] == 1)[0].tolist())
            hits.append(len(set(ranked.tolist()) & gold) / len(gold))
        out[f"recall@{k}"] = float(np.mean(hits)) if hits else float("nan")
    out["n_texts_with_gold"] = int(has_gold.sum())
    return out


def mean_average_precision(S, Y):
    """Per-text average precision over its candidate set, then averaged.

    Ranking metric, so it is independent of any threshold. That is why the
    paper uses it for model selection and early stopping.
    """
    aps = []
    for i in range(S.shape[0]):
        y = Y[i]
        if y.sum() == 0:
            continue
        s = np.nan_to_num(S[i], nan=-1.0)
        aps.append(average_precision_score(y, s))
    return float(np.mean(aps)) if aps else float("nan")


def evaluate(records, need_ids, threshold=0.5, per_need_thresholds=None,
             ks=(1, 5, 10, 20, 50), gold_map=None):
    """Full evaluation report.

    Args:
        records: iterable of (text_id, need_id, score, label). Only the pairs
            the system actually scored.
        need_ids: the full candidate need vocabulary.
        threshold: global decision threshold.
        per_need_thresholds: optional {need_id: threshold} override.
        gold_map: {text_id: iterable of need_id}. Supply this whenever a
            retriever gated which pairs were scored, so needs it never surfaced
            still count against recall. See _as_matrix.

    Returns a dict. `micro_macro_gap` is the number the paper says to read first.
    """
    text_ids, need_ids, S, Y = _as_matrix(records, need_ids, gold_map=gold_map)
    filled = np.nan_to_num(S, nan=-1.0)

    if per_need_thresholds:
        tvec = np.array([per_need_thresholds.get(n, threshold) for n in need_ids])
        pred = (filled >= tvec[None, :]).astype(int)
    else:
        pred = (filled >= threshold).astype(int)

    micro = precision_recall_fscore_support(
        Y.ravel(), pred.ravel(), average="binary", zero_division=0)
    macro_f1 = f1_score(Y, pred, average="macro", zero_division=0)

    # Per-need F1 restricted to needs that actually occur, so the macro average
    # is not diluted by thousands of needs with zero support in this split.
    support = Y.sum(axis=0)
    seen = np.where(support > 0)[0]
    per_need = {
        need_ids[j]: {
            "f1": float(f1_score(Y[:, j], pred[:, j], zero_division=0)),
            "support": int(support[j]),
        } for j in seen
    }
    macro_seen = (float(np.mean([v["f1"] for v in per_need.values()]))
                  if per_need else float("nan"))

    report = {
        "n_texts": len(text_ids),
        "n_needs": len(need_ids),
        "n_needs_with_support": int(len(seen)),
        "n_pairs_scored": int(np.isfinite(S).sum()),
        "threshold": threshold,
        "per_need_thresholds_used": bool(per_need_thresholds),
        "micro_precision": float(micro[0]),
        "micro_recall": float(micro[1]),
        "micro_f1": float(micro[2]),
        "macro_f1_all_needs": float(macro_f1),
        "macro_f1_needs_with_support": macro_seen,
        "micro_macro_gap": float(micro[2] - macro_seen) if per_need else float("nan"),
        "mAP": mean_average_precision(S, Y),
    }
    report.update(recall_at_k(S, Y, ks=ks))
    report["_per_need"] = per_need
    return report


def format_report(rep: dict) -> str:
    """Human-readable summary. Leads with the two numbers that matter."""
    lines = [
        "",
        "=" * 66,
        "  READ THESE FIRST",
        "=" * 66,
        f"  micro-F1                     {rep['micro_f1']:.4f}",
        f"  macro-F1 (needs w/ support)  {rep['macro_f1_needs_with_support']:.4f}",
        f"  GAP                          {rep['micro_macro_gap']:+.4f}"
        "   <- large gap = rare needs failing",
        f"  mAP                          {rep['mAP']:.4f}",
        "-" * 66,
    ]
    for k in sorted([k for k in rep if k.startswith("recall@")],
                    key=lambda s: int(s.split("@")[1])):
        lines.append(f"  {k:<28} {rep[k]:.4f}")
    lines += [
        "-" * 66,
        f"  micro precision / recall     {rep['micro_precision']:.4f} / "
        f"{rep['micro_recall']:.4f}",
        f"  threshold                    {rep['threshold']:.3f}"
        f"{'  (per-need overrides active)' if rep['per_need_thresholds_used'] else ''}",
        f"  texts / needs / scored pairs {rep['n_texts']} / {rep['n_needs']} / "
        f"{rep['n_pairs_scored']}",
        f"  needs with support           {rep['n_needs_with_support']}",
        "=" * 66,
    ]

    per_need = rep.get("_per_need") or {}
    if per_need:
        worst = sorted(per_need.items(), key=lambda kv: kv[1]["f1"])[:8]
        lines.append("  worst needs by F1 (read their justifications first):")
        for nid, v in worst:
            lines.append(f"    {nid:<22} F1={v['f1']:.3f}  support={v['support']}")
        lines.append("=" * 66)
    return "\n".join(lines)
