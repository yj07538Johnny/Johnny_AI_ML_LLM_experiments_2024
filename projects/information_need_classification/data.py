#!/usr/bin/env python3
"""Data layer for information-need classification.

Implements Sections 3 and 7.1 of the white paper:
  - the two input datasets and their expected schema
  - splitting by TEXT (never by pair, which leaks)
  - pair construction with hard + random negative sampling

Nothing here imports torch, so it can be exercised on its own.

Expected inputs (parquet or jsonl, auto-detected by extension):

  texts:  text_id  : str   unique
          text     : str   the tagged document
          need_ids : list[str]  gold information needs (may be empty)

  needs:  need_id       : str   unique
          need_label    : str   short human name
          justification : str   the prose that says what satisfies this need
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TEXT_COLUMNS = ("text_id", "text", "need_ids")
NEED_COLUMNS = ("need_id", "need_label", "justification")


# ------------------------------------------------------------------ load ----

def _read_any(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".jsonl", ".ndjson"):
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported extension {path.suffix!r} for {path}")


def _require(df: pd.DataFrame, cols, what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{what} is missing required column(s) {missing}. "
            f"Found: {list(df.columns)}")


def load_corpus(texts_path, needs_path):
    """Load and validate both datasets. Returns (texts_df, needs_df)."""
    texts = _read_any(texts_path)
    needs = _read_any(needs_path)
    _require(texts, TEXT_COLUMNS, f"texts file {texts_path}")
    _require(needs, NEED_COLUMNS, f"needs file {needs_path}")

    texts["need_ids"] = texts["need_ids"].apply(
        lambda v: [] if v is None else list(v))

    if texts["text_id"].duplicated().any():
        dupes = texts.loc[texts["text_id"].duplicated(), "text_id"].tolist()[:5]
        raise ValueError(f"duplicate text_id values, e.g. {dupes}")
    if needs["need_id"].duplicated().any():
        dupes = needs.loc[needs["need_id"].duplicated(), "need_id"].tolist()[:5]
        raise ValueError(f"duplicate need_id values, e.g. {dupes}")

    # A gold need that is not in the repository can never be scored. Fail loudly
    # instead of silently dropping it, which would quietly deflate recall.
    known = set(needs["need_id"])
    orphans = sorted({n for ids in texts["need_ids"] for n in ids} - known)
    if orphans:
        raise ValueError(
            f"{len(orphans)} gold need_id(s) are tagged on texts but absent from "
            f"the need repository, e.g. {orphans[:5]}. Every gold need must have "
            f"a justification or it cannot be scored.")

    return texts, needs


# ----------------------------------------------------------------- split ----

def split_by_text(texts: pd.DataFrame, val_frac=0.15, test_frac=0.15, seed=13):
    """Split by TEXT, never by pair.

    Section 2.1 of the paper. If the same text appears in training paired with
    need A and in validation paired with need B, the model has already read that
    text and the validation score is inflated. Splitting the texts first makes
    that impossible.
    """
    if not 0 <= val_frac + test_frac < 1:
        raise ValueError("val_frac + test_frac must be in [0, 1)")

    ids = texts["text_id"].tolist()
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_ids = set(ids[:n_test])
    val_ids = set(ids[n_test:n_test + n_val])

    def bucket(t):
        return "test" if t in test_ids else "val" if t in val_ids else "train"

    out = texts.copy()
    out["split"] = out["text_id"].map(bucket)
    return out


# ----------------------------------------------------------------- pairs ----

@dataclass
class Pair:
    text_id: str
    need_id: str
    text: str
    justification: str
    label: int
    source: str = "gold"       # gold | hard | random

    def as_dict(self):
        return {
            "text_id": self.text_id, "need_id": self.need_id,
            "text": self.text, "justification": self.justification,
            "label": self.label, "source": self.source,
        }


@dataclass
class SamplingPolicy:
    """Section 3.1. Ratios are per positive pair."""
    n_hard: int = 4
    n_random: int = 4
    seed: int = 13
    # Reported so a caller can see when the policy could not be satisfied
    # instead of assuming it was. Silent shortfall would misstate the ratio
    # the model actually trained on.
    stats: dict = field(default_factory=dict)


def build_pairs(texts: pd.DataFrame, needs: pd.DataFrame,
                policy: SamplingPolicy | None = None,
                hard_negatives: dict[str, list[str]] | None = None):
    """Construct training pairs from tagged texts + the need repository.

    Args:
        texts: rows with text_id, text, need_ids.
        needs: rows with need_id, justification.
        policy: sampling ratios. None means "positives only".
        hard_negatives: text_id -> ranked candidate need_ids from a retriever.
            Gold needs are filtered out here, so pass the raw ranking.
            Omit it and the sampler falls back to all-random, and says so.

    Returns:
        (list[Pair], stats dict)
    """
    policy = policy or SamplingPolicy(n_hard=0, n_random=0)
    just = dict(zip(needs["need_id"], needs["justification"]))
    all_need_ids = needs["need_id"].tolist()
    rng = random.Random(policy.seed)

    pairs: list[Pair] = []
    n_hard_wanted = n_hard_got = 0
    n_texts_without_hard = 0

    for row in texts.itertuples(index=False):
        gold = set(row.need_ids)
        for nid in row.need_ids:
            pairs.append(Pair(row.text_id, nid, row.text, just[nid], 1, "gold"))

        n_pos = len(gold)
        if n_pos == 0:
            continue

        # --- hard negatives: top-ranked non-gold candidates from the retriever
        want_hard = policy.n_hard * n_pos
        n_hard_wanted += want_hard
        ranked = (hard_negatives or {}).get(row.text_id, [])
        picked_hard = [n for n in ranked if n not in gold][:want_hard]
        if want_hard and not ranked:
            n_texts_without_hard += 1
        n_hard_got += len(picked_hard)
        for nid in picked_hard:
            pairs.append(Pair(row.text_id, nid, row.text, just[nid], 0, "hard"))

        # --- random negatives, plus any hard shortfall so the ratio is honest
        want_random = policy.n_random * n_pos + (want_hard - len(picked_hard))
        exclude = gold | set(picked_hard)
        pool = [n for n in all_need_ids if n not in exclude]
        take = min(want_random, len(pool))
        for nid in rng.sample(pool, take):
            pairs.append(Pair(row.text_id, nid, row.text, just[nid], 0, "random"))

    n_pos = sum(1 for p in pairs if p.label == 1)
    stats = {
        "n_pairs": len(pairs),
        "n_positive": n_pos,
        "n_negative": len(pairs) - n_pos,
        "n_hard": sum(1 for p in pairs if p.source == "hard"),
        "n_random": sum(1 for p in pairs if p.source == "random"),
        "hard_requested": n_hard_wanted,
        "hard_satisfied": n_hard_got,
        "texts_with_no_hard_candidates": n_texts_without_hard,
        "pos_neg_ratio": (len(pairs) - n_pos) / n_pos if n_pos else None,
    }
    if n_hard_wanted and n_hard_got < n_hard_wanted:
        stats["WARNING"] = (
            f"asked for {n_hard_wanted} hard negatives, got {n_hard_got}. "
            f"The shortfall was backfilled with random negatives, so the "
            f"effective policy is easier than {policy.n_hard} hard : "
            f"{policy.n_random} random. Run retriever.py to mine hard negatives.")
    policy.stats = stats
    return pairs, stats


def candidate_pairs(text_row, needs: pd.DataFrame, candidates=None):
    """Every (text, need) pair to be SCORED at evaluation or inference time.

    Unlike build_pairs this does no sampling: evaluation must see the whole
    candidate set, or the metrics describe a different task than the one served.
    """
    just = dict(zip(needs["need_id"], needs["justification"]))
    ids = candidates if candidates is not None else needs["need_id"].tolist()
    gold = set(getattr(text_row, "need_ids", []) or [])
    return [Pair(text_row.text_id, nid, text_row.text, just[nid],
                 int(nid in gold), "eval") for nid in ids]


def pairs_to_frame(pairs) -> pd.DataFrame:
    return pd.DataFrame([p.as_dict() for p in pairs])


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path
