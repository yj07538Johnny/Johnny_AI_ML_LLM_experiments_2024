"""
model_wrapper.py — THE ONLY FILE YOU EDIT to plug in your model.

The service (app.py), the client (client.py), and the container are all
model-agnostic. They only ever call `get_model().predict(texts, ...)`.

Default model: DistilBART-MNLI zero-shot classification (a transformer that
takes RAW TEXT and tokenizes internally). You give it text + candidate labels;
it returns the best label per row (and optionally the per-label scores).

"Pass the model as a file location":
  MODEL_PATH (env var) points to a local HuggingFace model directory
  (the output of `AutoModel.from_pretrained(...).save_pretrained(dir)`, or a
  downloaded snapshot). Mount that directory into the container and point
  MODEL_PATH at it. It may also be a hub id (e.g. "valhalla/distilbart-mnli-12-3").

The pipeline is built once, lazily, on first request and cached for the life
of the process (the model is NOT reloaded per call).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

# ---------------------------------------------------------------------------
# Configuration — all overridable by environment variable (docker -e / compose)
# ---------------------------------------------------------------------------
# Where the model lives. A local directory (recommended: mount it) or a hub id.
MODEL_PATH = os.environ.get("MODEL_PATH", "model_artifacts/distilbart-mnli")
# HF task. Zero-shot for MNLI models; "text-classification" for a fine-tuned head.
MODEL_TASK = os.environ.get("MODEL_TASK", "zero-shot-classification")
# Device: "auto" -> first visible CUDA GPU else CPU. Or an int index / "cpu".
# CUDA_VISIBLE_DEVICES already excludes the display GPU on this host.
MODEL_DEVICE = os.environ.get("MODEL_DEVICE", "auto")
# Default candidate labels for zero-shot (comma-separated). A request may override.
_DEFAULT_LABELS = os.environ.get("MODEL_CANDIDATE_LABELS", "")
DEFAULT_CANDIDATE_LABELS = [s.strip() for s in _DEFAULT_LABELS.split(",") if s.strip()]
# Zero-shot hypothesis template. "{}" is filled with each candidate label.
HYPOTHESIS_TEMPLATE = os.environ.get("MODEL_HYPOTHESIS_TEMPLATE", "This example is {}.")
# Allow more than one label to be true (independent sigmoids) vs. softmax over labels.
MULTI_LABEL = os.environ.get("MODEL_MULTI_LABEL", "false").lower() in ("1", "true", "yes")
# Inference batch size handed to the HF pipeline.
BATCH_SIZE = int(os.environ.get("MODEL_BATCH_SIZE", "16"))


def _resolve_device(spec: str) -> int:
    """HF pipeline device: -1 = CPU, >=0 = CUDA index within CUDA_VISIBLE_DEVICES."""
    if spec == "auto":
        try:
            import torch
            return 0 if torch.cuda.is_available() else -1
        except Exception:
            return -1
    if spec == "cpu":
        return -1
    return int(spec)


class ModelWrapper:
    """Adapter around a HuggingFace pipeline. Adapt `predict` if your task differs."""

    def __init__(self, pipe, task: str):
        self._pipe = pipe
        self.task = task

    # ---- required ---------------------------------------------------------
    def predict(
        self,
        texts: Sequence[str],
        candidate_labels: Optional[Sequence[str]] = None,
        hypothesis_template: Optional[str] = None,
        return_scores: bool = False,
    ):
        """Return one label per text. If return_scores, also return per-row
        {label: score} dicts. `candidate_labels` is required for zero-shot
        (falls back to MODEL_CANDIDATE_LABELS env)."""
        texts = [_as_text(t) for t in texts]

        if self.task == "zero-shot-classification":
            labels = list(candidate_labels or DEFAULT_CANDIDATE_LABELS)
            if not labels:
                raise ValueError(
                    "zero-shot requires candidate_labels (per request) or the "
                    "MODEL_CANDIDATE_LABELS env var."
                )
            results = self._pipe(
                texts,
                candidate_labels=labels,
                hypothesis_template=hypothesis_template or HYPOTHESIS_TEMPLATE,
                multi_label=MULTI_LABEL,
                batch_size=BATCH_SIZE,
            )
            if isinstance(results, dict):          # single-text -> single dict
                results = [results]
            preds = [r["labels"][0] for r in results]
            if return_scores:
                scores = [dict(zip(r["labels"], (float(s) for s in r["scores"])))
                          for r in results]
                return preds, scores
            return preds

        # Plain text-classification (fine-tuned head): no candidate labels.
        results = self._pipe(texts, batch_size=BATCH_SIZE, truncation=True)
        if isinstance(results, dict):
            results = [results]
        preds = [r["label"] for r in results]
        if return_scores:
            scores = [{r["label"]: float(r["score"])} for r in results]
            return preds, scores
        return preds


# ---------------------------------------------------------------------------
# Loading — isolated so the rest of the service never imports transformers.
# ---------------------------------------------------------------------------
def load_pipeline():
    """Build the HuggingFace pipeline from MODEL_PATH.

    >>> EDIT here only if your model needs a non-standard loader. <<<
    """
    # A local path must exist; a bare hub id (no slash-path on disk) is allowed
    # so you can test without pre-downloading.
    if os.path.sep in MODEL_PATH and not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No model at {MODEL_PATH!r}. Set MODEL_PATH to your model directory "
            f"or mount it into the container at that path."
        )
    from transformers import pipeline
    return pipeline(MODEL_TASK, model=MODEL_PATH, device=_resolve_device(MODEL_DEVICE))


@lru_cache(maxsize=1)
def get_model() -> ModelWrapper:
    """Load once, cache for the process lifetime (not reloaded per request)."""
    return ModelWrapper(load_pipeline(), task=MODEL_TASK)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))
