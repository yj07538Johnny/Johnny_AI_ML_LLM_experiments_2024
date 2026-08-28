#!/usr/bin/env python3
"""Generate a synthetic corpus so the pipeline runs end to end before real data.

This exists to exercise the CODE, not to produce a result. The texts are
template-assembled from need-specific vocabulary, so the task is far easier than
the real one and any score it produces is meaningless. What it does prove is
that pair construction, negative sampling, training, threshold tuning and
evaluation all execute and agree on schemas.

It deliberately reproduces two properties of the real problem that break naive
implementations:

  - a LONG TAIL of needs. Most needs have a handful of texts, a few have many.
    This is what separates micro-F1 from macro-F1.
  - MULTI-LABEL texts. Many texts satisfy two or three needs at once.

    python make_synthetic_corpus.py --out corpus --n-texts 600 --n-needs 120
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

TOPICS = [
    ("coastal adaptation", ["sea level", "surge boundary", "shoreline", "levee",
                            "tidal flooding", "coastal erosion"]),
    ("municipal finance", ["municipal bond", "tax levy", "budget shortfall",
                           "debt issuance", "appropriation", "bond rating"]),
    ("zoning policy", ["zoning variance", "setback", "density limit",
                       "land use", "permitting", "overlay district"]),
    ("port logistics", ["berth", "container volume", "throughput", "drayage",
                        "terminal capacity", "harbor dredging"]),
    ("public transit", ["ridership", "headway", "fare recovery", "bus rapid",
                        "rail corridor", "service span"]),
    ("water systems", ["aquifer", "treatment plant", "lead service line",
                       "stormwater", "potable supply", "watershed"]),
    ("emergency management", ["evacuation route", "shelter capacity",
                              "incident command", "mutual aid", "hazard plan"]),
    ("housing supply", ["affordable units", "vacancy rate", "rent burden",
                        "inclusionary", "housing starts", "displacement"]),
    ("energy grid", ["substation", "peak demand", "interconnection",
                     "transmission", "outage", "distributed generation"]),
    ("air quality", ["particulate", "monitoring station", "ozone exceedance",
                     "emissions inventory", "attainment"]),
]

FRAMES = [
    "The {a} report described {p}, with attention to {q}.",
    "Officials reviewed {p} and noted that {q} remained unresolved.",
    "A {a} briefing summarised {p}; the record also covers {q}.",
    "Testimony addressed {p}, and separately documented {q}.",
    "The filing sets out {p}. It further describes {q}.",
]

QUALIFIERS = ["quarterly", "interim", "annual", "technical", "preliminary",
              "consolidated", "field", "oversight"]

JUST_FRAMES = [
    "Analysts need documents describing {p}, including {q} and any supporting "
    "figures.",
    "This need covers material on {p}. Records that quantify {q} are in scope.",
    "Documents are responsive when they report {p} or characterise {q}.",
]


def build(n_texts, n_needs, seed=13, max_needs_per_text=3):
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)

    # ---- needs, each anchored to a topic and two of its terms
    needs = []
    for i in range(n_needs):
        topic, terms = TOPICS[i % len(TOPICS)]
        p, q = rng.sample(terms, 2)
        needs.append({
            "need_id": f"IN-{i:04d}",
            "need_label": f"{topic}: {p}",
            "justification": rng.choice(JUST_FRAMES).format(p=p, q=q),
            "_terms": (p, q),
        })
    needs_df = pd.DataFrame(needs)

    # ---- a Zipf-ish popularity curve, so the tail is real
    weights = 1.0 / np.arange(1, n_needs + 1) ** 1.1
    weights = weights / weights.sum()
    order = nprng.permutation(n_needs)
    weights = weights[np.argsort(order)]

    rows = []
    for t in range(n_texts):
        k = int(nprng.choice([1, 1, 1, 2, 2, 3]))
        k = min(k, max_needs_per_text)
        picked = nprng.choice(n_needs, size=k, replace=False, p=weights)
        sentences = []
        for j in picked:
            p, q = needs[j]["_terms"]
            sentences.append(rng.choice(FRAMES).format(
                a=rng.choice(QUALIFIERS), p=p, q=q))
        # a distractor sentence from an unrelated topic, so the text is not a
        # pure concatenation of its own labels
        dtopic, dterms = rng.choice(TOPICS)
        sentences.append(f"Unrelated background mentions {rng.choice(dterms)}.")
        rng.shuffle(sentences)
        rows.append({
            "text_id": f"DOC-{t:05d}",
            "text": " ".join(sentences),
            "need_ids": [needs[j]["need_id"] for j in sorted(picked)],
        })

    return pd.DataFrame(rows), needs_df.drop(columns=["_terms"])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--n-texts", type=int, default=600)
    ap.add_argument("--n-needs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    texts, needs = build(args.n_texts, args.n_needs, seed=args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    texts.to_parquet(out / "texts.parquet", index=False)
    needs.to_parquet(out / "needs.parquet", index=False)

    counts = pd.Series([n for ids in texts["need_ids"] for n in ids]).value_counts()
    print(f"  {len(texts)} texts -> {out / 'texts.parquet'}")
    print(f"  {len(needs)} needs -> {out / 'needs.parquet'}")
    print(f"  labels per text: mean "
          f"{np.mean([len(i) for i in texts['need_ids']]):.2f}")
    print(f"  need support: max={counts.max()} median={int(counts.median())} "
          f"min={counts.min()}  needs never used="
          f"{len(needs) - counts.size}")
    print(f"  needs with support < 5: {(counts < 5).sum()}  "
          f"(this is the tail macro-F1 exposes)")
    print("\n  SYNTHETIC DATA. Scores from this corpus measure the code, "
          "not the method.")


if __name__ == "__main__":
    main()
