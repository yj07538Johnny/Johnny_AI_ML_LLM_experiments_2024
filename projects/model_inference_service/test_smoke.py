"""
test_smoke.py — end-to-end check against a RUNNING service.

    # terminal 1
    docker compose up --build
    # terminal 2
    python test_smoke.py            # or: BASE_URL=http://host:8000 python test_smoke.py

Exercises /health, /predict (texts), JSON dataframe, and Parquet dataframe.
Uses zero-shot labels ["positive","negative"] — adjust for your model.
"""

import os
import sys

import pandas as pd

from client import InferenceClient

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
LABELS = os.environ.get("SMOKE_LABELS", "positive,negative").split(",")


def main() -> int:
    clf = InferenceClient(BASE_URL)

    health = clf.health()
    print("health:", health)
    if not health.get("model_loaded"):
        print("!! model not loaded:", health.get("error"))
        return 1

    texts = ["this was an excellent experience", "cold food and rude staff"]
    print("texts :", clf.predict_texts(texts, candidate_labels=LABELS))

    df = pd.DataFrame({"review": texts})

    out_json = clf.predict_dataframe(df, target_column="review",
                                     prediction_column="label",
                                     candidate_labels=LABELS, transport="json")
    print("json  :\n", out_json)
    assert "label" in out_json.columns and len(out_json) == len(df)

    out_pq = clf.predict_dataframe(df, target_column="review",
                                   prediction_column="label",
                                   candidate_labels=LABELS, transport="parquet")
    print("parquet:\n", out_pq)
    assert "label" in out_pq.columns and len(out_pq) == len(df)
    assert list(out_pq.index) == list(df.index)

    print("\nOK — all smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
