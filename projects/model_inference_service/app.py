"""
app.py — network-facing model-inference microservice (FastAPI).

Generic contract: give it a dataframe (as Parquet bytes or JSON records), tell
it which column holds the text to read and what to name the prediction column,
and get the dataframe back with the prediction column added.

Endpoints
---------
GET  /health                 liveness + whether the model loaded
POST /predict                a list of texts (one column) -> predictions
POST /predict/dataframe      JSON records + column names  -> records + prediction col
POST /predict/parquet        Parquet body + column names  -> Parquet body + prediction col

The Parquet path is the one to use for large frames or embedding/list columns:
dtypes and array columns survive the round trip; JSON is for small ad-hoc calls.
Zero-shot models need `candidate_labels` (per request, or the MODEL_CANDIDATE_LABELS
env default).
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from model_wrapper import MODEL_PATH, MODEL_TASK, get_model

app = FastAPI(
    title="Model Inference Service",
    version="1.0.0",
    description="Pass a dataframe + read/write column names; get predictions back. "
                "Model is loaded from MODEL_PATH. Default: DistilBART-MNLI zero-shot.",
)

PARQUET_MEDIA = "application/vnd.apache.parquet"


# ---------------------------------------------------------------------------
# Schemas (JSON transport)
# ---------------------------------------------------------------------------
class TextsRequest(BaseModel):
    texts: List[str] = Field(..., description="One column's worth of texts.")
    candidate_labels: Optional[List[str]] = Field(
        None, description="Zero-shot label set (required for MNLI models)."
    )
    hypothesis_template: Optional[str] = None
    return_scores: bool = False


class TextsResponse(BaseModel):
    predictions: List[Any]
    scores: Optional[List[Dict[str, float]]] = None


class DataFrameRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., description="df.to_dict('records').")
    target_column: str = Field(..., description="Column holding the text to read.")
    prediction_column: str = Field("prediction", description="Column to write.")
    candidate_labels: Optional[List[str]] = None
    hypothesis_template: Optional[str] = None
    return_scores: bool = False
    score_column: str = Field("scores", description="Column for per-label scores.")


class DataFrameResponse(BaseModel):
    records: List[Dict[str, Any]]
    target_column: str
    prediction_column: str
    n_rows: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    loaded, error = True, None
    try:
        get_model()
    except Exception as exc:
        loaded, error = False, str(exc)
    return {"status": "ok", "model_loaded": loaded, "model_path": MODEL_PATH,
            "task": MODEL_TASK, "error": error}


@app.post("/predict", response_model=TextsResponse)
def predict(req: TextsRequest) -> TextsResponse:
    if not req.texts:
        raise HTTPException(400, "texts must not be empty.")
    model = _model_or_503()
    out = model.predict(req.texts, candidate_labels=req.candidate_labels,
                        hypothesis_template=req.hypothesis_template,
                        return_scores=req.return_scores)
    preds, scores = out if req.return_scores else (out, None)
    return TextsResponse(predictions=preds, scores=scores)


@app.post("/predict/dataframe", response_model=DataFrameResponse)
def predict_dataframe(req: DataFrameRequest) -> DataFrameResponse:
    if not req.records:
        raise HTTPException(400, "records must not be empty.")
    if req.target_column not in req.records[0]:
        raise HTTPException(
            400, f"target_column {req.target_column!r} not in records. "
                 f"Available: {sorted(req.records[0].keys())}")

    texts = [row.get(req.target_column) for row in req.records]
    model = _model_or_503()
    out = model.predict(texts, candidate_labels=req.candidate_labels,
                        hypothesis_template=req.hypothesis_template,
                        return_scores=req.return_scores)
    preds, scores = out if req.return_scores else (out, None)

    result = []
    for i, row in enumerate(req.records):
        new = dict(row)                       # does not mutate the input
        new[req.prediction_column] = preds[i]
        if scores is not None:
            new[req.score_column] = scores[i]
        result.append(new)
    return DataFrameResponse(records=result, target_column=req.target_column,
                             prediction_column=req.prediction_column, n_rows=len(result))


@app.post("/predict/parquet")
async def predict_parquet(
    request: Request,
    target_column: str = Query(..., description="Column holding the text to read."),
    prediction_column: str = Query("prediction", description="Column to write."),
    candidate_labels: Optional[str] = Query(
        None, description="Comma-separated zero-shot labels."),
    hypothesis_template: Optional[str] = Query(None),
    return_scores: bool = Query(False),
    score_column: str = Query("scores"),
) -> Response:
    """Body: a Parquet-serialized dataframe. Returns: Parquet with prediction col."""
    import pandas as pd

    body = await request.body()
    if not body:
        raise HTTPException(400, "empty body — send a Parquet-serialized dataframe.")
    try:
        df = pd.read_parquet(io.BytesIO(body))
    except Exception as exc:
        raise HTTPException(400, f"could not read Parquet body: {exc}")
    if target_column not in df.columns:
        raise HTTPException(
            400, f"target_column {target_column!r} not in dataframe. "
                 f"Columns: {list(df.columns)}")

    labels = [s.strip() for s in candidate_labels.split(",")] if candidate_labels else None
    model = _model_or_503()
    out = model.predict(df[target_column].tolist(), candidate_labels=labels,
                        hypothesis_template=hypothesis_template,
                        return_scores=return_scores)
    preds, scores = out if return_scores else (out, None)

    df = df.copy()
    df[prediction_column] = preds
    if scores is not None:
        df[score_column] = scores

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return Response(content=buf.getvalue(), media_type=PARQUET_MEDIA,
                    headers={"X-Rows": str(len(df)),
                             "X-Prediction-Column": prediction_column})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _model_or_503():
    try:
        return get_model()
    except Exception as exc:
        raise HTTPException(503, f"Model unavailable: {exc}")
