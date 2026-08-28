# Information-need classification

Reference implementation for
[`papers/information-need-cross-encoder`](../../papers/information-need-cross-encoder):
predict which information needs a report satisfies by scoring
`(report, need justification)` pairs with a DistilBERT-MNLI cross-encoder.

Read the paper first. This directory is the method made executable, and the
design decisions are argued there rather than here.

| File | Role |
|---|---|
| `gpu.py` | Pins the process to one GPU before torch is imported |
| `data.py` | Schema, split-by-report, pair construction, negative sampling |
| `retriever.py` | Frozen bi-encoder. Mines hard negatives, and is serving stage one |
| `metrics.py` | micro/macro F1 and the gap, mAP, recall@k, threshold tuning |
| `train_cross_encoder.py` | The training loop |
| `evaluate.py` | Held-out scoring against the split recorded at training time |
| `test_metrics.py` | 26 checks against hand-computable cases |
| `make_synthetic_corpus.py` | Fake corpus, so the pipeline runs without real data |

## Input schema

Two files, parquet or jsonl.

**reports**

| column | type | meaning |
|---|---|---|
| `text_id` | str, unique | report identifier |
| `text` | str | the report text or abstract |
| `need_ids` | list[str] | needs this report was cited against, may be empty |

**needs**

| column | type | meaning |
|---|---|---|
| `need_id` | str, unique | need identifier |
| `need_label` | str | short human name |
| `justification` | str | prose saying what would satisfy this need |

`load_corpus` fails loudly if a cited `need_id` has no justification, since such
a need can never be scored and dropping it silently would deflate recall.

## Run it

```bash
pip install torch transformers pandas pyarrow scikit-learn

# optional: synthetic corpus, to prove the pipeline runs
python make_synthetic_corpus.py --out corpus --n-texts 600 --n-needs 120

# 1. mine hard negatives and measure the stage-one recall ceiling
python retriever.py --texts corpus/texts.parquet --needs corpus/needs.parquet \
    --out corpus/hard_negatives.json --top-k 50

# 2. train
python train_cross_encoder.py --texts corpus/texts.parquet \
    --needs corpus/needs.parquet \
    --hard-negatives corpus/hard_negatives.json --out runs/v1

# 3. evaluate on the held-out split
python evaluate.py --run runs/v1 --texts corpus/texts.parquet \
    --needs corpus/needs.parquet --hard-negatives corpus/hard_negatives.json

# tests need no GPU and no model download
python test_metrics.py
```

## Defaults

| Setting | Value |
|---|---|
| base checkpoint | `typeform/distilbert-base-uncased-mnli` |
| retriever | `sentence-transformers/all-MiniLM-L6-v2`, frozen |
| max sequence length | 256, `truncation="only_first"` |
| batch size | 32 |
| optimiser | AdamW, weight decay 0.01, none on bias/LayerNorm |
| learning rate | 2e-5, linear decay, 10% warmup |
| epochs | 3, early stopping on validation mAP |
| loss | `BCEWithLogitsLoss` on a single output unit |
| negative sampling | 4 hard + 4 random per positive |
| precision | fp16 autocast on CUDA, `--no-fp16` to disable |

## Five things the code enforces that are easy to get wrong

**One GPU, pinned before torch loads.** With more than one device visible and no
distributed launcher, several training paths auto-wrap in `nn.DataParallel`,
which on Pascal-era cards can stall rather than fail. `gpu.py` sets a single
device first. Override with `IN_CLS_GPU=3`, or `IN_CLS_GPU=cpu`. The default
device index is specific to the machine this was written on, so change it.

**Split by report, never by pair.** If a report appears in training paired with
need A and in validation paired with need B, the model has already read it and
the validation number is inflated. `split_by_text` runs before any pair is
built, and `evaluate.py` refuses to run without the `splits.json` written at
training time.

**The negative sampling ratio is reported, not assumed.** When the retriever
cannot supply enough hard negatives the shortfall is backfilled with random
ones, which makes the effective policy easier than requested. `build_pairs`
returns `hard_requested` against `hard_satisfied` and warns when they differ.

**Evaluation scores the shortlist, because that is what serving does.** A cited
need outside the retriever's top-k is deliberately not scored, since the served
system would never score it either. It is carried in `gold_map` and counts as a
miss. Adding it to the candidate set would let the cross-encoder recover a need
the retriever never surfaced, and reported recall would then exceed the
retriever's own ceiling, which production cannot do.

**Model selection uses mAP, not F1.** mAP evaluates the ranking, so it does not
depend on a threshold nobody has tuned yet. Thresholds are fitted afterwards, on
validation, with a per-need override only where support clears
`--min-support-per-need` (default 10). A threshold fitted on three examples is
noise.

## Reading the output

`evaluate.py` leads with the two numbers the paper argues matter most.

- **The micro-F1 to macro-F1 gap.** Micro pools every pair decision, so frequent
  needs dominate it. Macro averages per-need F1, so a need with six examples
  counts as much as one with nine hundred. A large gap means the model works on
  common needs and fails on rare ones.
- **recall@k.** A hard ceiling on the whole system. If the right need is not in
  the top k, no amount of cross-encoder quality recovers it. `retriever.py`
  prints this before training, so the ceiling is known in advance.

The report also lists the eight worst needs by F1. Read their justifications
before concluding anything about the model. A need described in one vague
sentence will score badly no matter how long it trains, and that is a data
problem with a data fix.

## Limits

`make_synthetic_corpus.py` assembles reports out of each need's own vocabulary,
so the synthetic task is far easier than a real one. It proves the code runs. It
proves nothing about the method, and the script says so on every invocation.
Delete it if it is in the way; nothing else depends on it.

Sigmoid outputs are uncalibrated. They rank correctly but should not be read as
probabilities without a reliability check.

## License

GPL-3.0, per the repository root.
