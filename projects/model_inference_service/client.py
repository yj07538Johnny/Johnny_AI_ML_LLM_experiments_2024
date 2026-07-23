"""
client.py — reusable client for the model-inference microservice.

Import this in your own code and work in terms of dataframes and columns; the
HTTP details (and Parquet transport) are hidden.

    from client import InferenceClient

    clf = InferenceClient("http://localhost:8000")

    # a dataframe (or any slice) -> a NEW df with a prediction column added.
    # You pick the text column to read and the column name to write.
    out = clf.predict_dataframe(
        df, target_column="sentence", prediction_column="label",
        candidate_labels=["entailment", "contradiction", "neutral"],
    )

    # want the per-label scores too?
    out = clf.predict_dataframe(df, target_column="sentence",
                                candidate_labels=[...], return_scores=True)

    # a plain list of texts
    clf.predict_texts(["a sentence", "another"], candidate_labels=[...])

By default `predict_dataframe` uses Parquet transport (fast, dtype-preserving,
handles large frames and array columns). Pass transport="json" for small calls.
"""

from __future__ import annotations

import io
from typing import Any, List, Optional, Sequence

import requests

PARQUET_MEDIA = "application/vnd.apache.parquet"


class InferenceClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- a list of texts ---------------------------------------------------
    def predict_texts(self, texts: Sequence[str],
                      candidate_labels: Optional[Sequence[str]] = None,
                      hypothesis_template: Optional[str] = None,
                      return_scores: bool = False):
        payload = {"texts": list(texts), "return_scores": return_scores}
        if candidate_labels is not None:
            payload["candidate_labels"] = list(candidate_labels)
        if hypothesis_template is not None:
            payload["hypothesis_template"] = hypothesis_template
        r = self._post_json("/predict", payload)
        return (r["predictions"], r.get("scores")) if return_scores else r["predictions"]

    # -- a dataframe or a slice of one ------------------------------------
    def predict_dataframe(self, df, target_column: str,
                          prediction_column: str = "prediction",
                          candidate_labels: Optional[Sequence[str]] = None,
                          hypothesis_template: Optional[str] = None,
                          return_scores: bool = False,
                          score_column: str = "scores",
                          transport: str = "parquet"):
        """Return a COPY of `df` with `prediction_column` added.

        `df` may be a whole dataframe or any slice (df.iloc[...], a boolean
        mask, a single-row frame, ...). The original index is preserved.
        """
        import pandas as pd

        if target_column not in df.columns:
            raise KeyError(f"target_column {target_column!r} not in dataframe. "
                           f"Columns: {list(df.columns)}")

        if transport == "parquet":
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            params = {"target_column": target_column,
                      "prediction_column": prediction_column,
                      "return_scores": str(return_scores).lower(),
                      "score_column": score_column}
            if candidate_labels is not None:
                params["candidate_labels"] = ",".join(candidate_labels)
            if hypothesis_template is not None:
                params["hypothesis_template"] = hypothesis_template
            resp = requests.post(f"{self.base_url}/predict/parquet",
                                 params=params, data=buf.getvalue(),
                                 headers={"Content-Type": PARQUET_MEDIA},
                                 timeout=self.timeout)
            if not resp.ok:
                raise RuntimeError(f"{resp.status_code} from /predict/parquet: "
                                   f"{_safe_detail(resp)}")
            out = pd.read_parquet(io.BytesIO(resp.content))
            out.index = df.index
            return out

        # JSON transport
        payload = {"records": df.to_dict(orient="records"),
                   "target_column": target_column,
                   "prediction_column": prediction_column,
                   "return_scores": return_scores,
                   "score_column": score_column}
        if candidate_labels is not None:
            payload["candidate_labels"] = list(candidate_labels)
        if hypothesis_template is not None:
            payload["hypothesis_template"] = hypothesis_template
        r = self._post_json("/predict/dataframe", payload)
        return pd.DataFrame(r["records"], index=df.index)

    def health(self) -> dict:
        return requests.get(f"{self.base_url}/health", timeout=self.timeout).json()

    # -- internals ---------------------------------------------------------
    def _post_json(self, path: str, payload: dict) -> dict:
        resp = requests.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(f"{resp.status_code} from {path}: {_safe_detail(resp)}")
        return resp.json()


def _safe_detail(resp) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text


if __name__ == "__main__":
    clf = InferenceClient()
    print("health:", clf.health())
    print(clf.predict_texts(
        ["the service worked on the first try"],
        candidate_labels=["positive", "negative"]))
