# Model Inference Service

A generic, reusable Dockerized microservice that runs an **already-trained**
model over a column of a dataframe. Pass it a dataframe plus the column to
**read** and the column to **write**; it runs the model and returns the
dataframe with the prediction column added.

Default model: **DistilBART-MNLI** zero-shot classification (a transformer that
takes raw text and tokenizes internally). Swap in any model by editing one file.

You provide the model as a **file location** — set `MODEL_PATH` (env var) and
mount your model directory into the container. No rebuild to swap models.

## Files

| File | Purpose |
|------|---------|
| `model_wrapper.py` | **The only file you edit.** Loads the model from `MODEL_PATH` and defines `predict`. |
| `app.py` | FastAPI service — JSON and Parquet endpoints. |
| `client.py` | Reusable Python client — work in dataframes and columns. |
| `Dockerfile`, `docker-compose.yml` | Containerize and run. |
| `test_smoke.py` | End-to-end check against a running service. |
| `requirements.txt` | Dependencies. |

## 1. Provide the model (a file location)

`MODEL_PATH` points at a local HuggingFace model directory (the output of
`model.save_pretrained(dir)` / a downloaded snapshot), or a hub id for testing.

Get DistilBART-MNLI onto disk once:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
name = "valhalla/distilbart-mnli-12-3"
d = "model_artifacts/distilbart-mnli"
AutoTokenizer.from_pretrained(name).save_pretrained(d)
AutoModelForSequenceClassification.from_pretrained(name).save_pretrained(d)
```

`docker-compose.yml` bind-mounts `./model_artifacts/distilbart-mnli` to
`/models/distilbart-mnli` (where `MODEL_PATH` points). To use a different model,
change `MODEL_PATH` + the mount — no rebuild.

### Plugging in a different model

Edit `model_wrapper.py`:
- **Another zero-shot / text-classification transformer:** just change
  `MODEL_PATH` (and `MODEL_TASK` if it has a fine-tuned head).
- **A non-transformers model (sklearn, a torch head over embeddings, ...):**
  adapt `load_pipeline()` and `ModelWrapper.predict()`. Everything else is
  model-agnostic and only calls `get_model().predict(...)`.

## 2. Run it

```bash
docker compose up --build          # serves on http://localhost:8000
```

Without Docker:

```bash
pip install -r requirements.txt
MODEL_PATH=model_artifacts/distilbart-mnli uvicorn app:app --port 8000
```

Interactive API docs: `http://localhost:8000/docs`.

## 3. Call it (the reusable client)

```python
import pandas as pd
from client import InferenceClient

clf = InferenceClient("http://localhost:8000")
df = pd.DataFrame({"sentence": ["the plan worked", "everything broke"]})

# read the "sentence" column, write a "label" column -> a NEW df
out = clf.predict_dataframe(
    df,
    target_column="sentence",
    prediction_column="label",
    candidate_labels=["success", "failure"],   # zero-shot label set
)

# per-label scores too
out = clf.predict_dataframe(df, target_column="sentence",
                            candidate_labels=["success", "failure"],
                            return_scores=True)

# a slice works the same way; the index is preserved
clf.predict_dataframe(df.iloc[0:1], target_column="sentence",
                      candidate_labels=["success", "failure"])
```

`predict_dataframe` defaults to **Parquet transport** (fast, dtype-preserving,
handles large frames and array/embedding columns). Pass `transport="json"` for
small ad-hoc calls.

## Endpoints

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/health` | — | status, model_loaded, model_path, task |
| POST | `/predict` | `{texts[], candidate_labels[], return_scores?}` | `{predictions[], scores?}` |
| POST | `/predict/dataframe` | `{records[], target_column, prediction_column?, candidate_labels[], return_scores?}` | `{records[], ...}` |
| POST | `/predict/parquet` | Parquet body; params in query string | Parquet body with prediction column |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "texts": ["shipped on time", "arrived broken"],
  "candidate_labels": ["positive", "negative"]
}'
```

## Configuration (environment variables)

| Var | Default | Meaning |
|-----|---------|---------|
| `MODEL_PATH` | `/models/distilbart-mnli` | Model file location (dir or hub id). |
| `MODEL_TASK` | `zero-shot-classification` | HF task; `text-classification` for a fine-tuned head. |
| `MODEL_CANDIDATE_LABELS` | `""` | Default zero-shot labels (comma-separated); a request may override. |
| `MODEL_HYPOTHESIS_TEMPLATE` | `This example is {}.` | Zero-shot hypothesis template. |
| `MODEL_MULTI_LABEL` | `false` | Independent per-label scores vs. softmax over labels. |
| `MODEL_DEVICE` | `auto` | `auto` (GPU if visible else CPU), `cpu`, or a CUDA index. |
| `MODEL_BATCH_SIZE` | `16` | Inference batch size. |

## Notes

- **Model loading** is isolated in `model_wrapper.py` and cached for the process
  lifetime — one copy in memory, not reloaded per call.
- **Missing model** doesn't crash the server — `/health` reports it and
  prediction calls return `503` with a clear message.
- **GPU:** the base image is CPU-only. For P100 inference, switch to a CUDA base
  image and uncomment the GPU block in `docker-compose.yml`.
- **Scaling:** one worker holds one copy of the model. Add containers behind a
  load balancer, or raise `--workers`, once you know the memory footprint.
