# Linking Reports to Information Needs with a Cross-Encoder

Supervised learning with pairwise attention over reports and need justifications,
trained on author citations.

**Johnny Morgan**, University of Maryland, Baltimore County

| File | What |
|---|---|
| `main.pdf` | The paper, 22 pages, 15 sections |
| `main.tex` | LaTeX source |
| `Information_Need_Cross_Encoder_Briefing.pptx` | 29-slide deck, speaker notes on every slide |
| `figures/` | 13 figures, 200 dpi |
| `make_figures.py` | Regenerates every figure |
| `make_deck.py` | Regenerates the deck from those figures |

Reference implementation of the training and evaluation method:
[`projects/information_need_classification`](../../projects/information_need_classification).

## The problem

Given a corpus of reports and a repository of information needs, each need
carrying a written justification, predict which needs a new report satisfies.
The labels come from authors: when an author cites a report as satisfying a
need, that citation is a training example.

The working scale for this paper is roughly 20,000 reports against more than
1,000 needs, so about 20 million candidate report-need pairs at a citation
density near 0.15%.

## The method

Concatenate the report and one need's justification into a single sequence:

```
[CLS]  <report text>  [SEP]  <need justification>  [SEP]
```

Encode it with DistilBERT-MNLI and read one sigmoid output. Because both
segments live in one sequence, self-attention runs across the boundary at every
layer, so tokens in the report attend directly to tokens in the justification.
That is what "pairwise attention" means here, and it is what a bi-encoder cannot
do once it has compressed each side to a single vector.

Two properties follow from putting the label on the *pair* rather than the
report. The justification becomes half the model input rather than an unused
column. And a need that opens tomorrow can be scored as soon as someone writes
its justification, with no retraining and no new output unit.

MNLI pretraining transfers because the questions have the same shape. MNLI asks
whether a premise supports a hypothesis; we ask whether a report satisfies an
information need.

## What makes this harder than it looks

Citations give positives and never negatives. An uncited report-need pair may be
irrelevant, or relevant with a different report cited instead, or relevant and
simply never written down. Nothing in the data separates those cases, which puts
the problem in the positive-unlabeled setting.

Three consequences drive most of the paper:

- Every negative used in training is an assumption, and the assumption is
  weakest exactly where hard negative mining looks hardest.
- Measured precision is a floor rather than an estimate, because a correct
  prediction on an uncited pair scores as a false positive.
- The apparent errors are the deliverable. A high-scoring uncited pair is the
  relationship the system exists to surface.

Sections 4, 6.2 and 13 deal with those in turn, including a discovery loop that
converts model output plus human adjudication into labels that did not exist
before.

## Rebuilding

```bash
pip install matplotlib python-pptx
python make_figures.py          # writes figures/
python make_deck.py             # writes the .pptx from figures/
pdflatex main && pdflatex main  # twice, for the table of contents
```

Both scripts are deterministic and take `--out` if you want the output elsewhere.

## Figures

| | |
|---|---|
| 01 | Supervised learning, one training step |
| 02 | Self-attention: query, key, value |
| 03 | The attention matrix inside a cross-encoder, with the cross-segment blocks marked |
| 04 | Cross-encoder against bi-encoder, and what each costs |
| 05 | Where DistilBERT-MNLI comes from |
| 06 | Turning two datasets into training pairs |
| 07 | The training pipeline end to end |
| 08 | Multi-label evaluation and thresholds |
| 09 | Two-stage serving |
| 10 | One report scored against three candidate needs |
| 11 | What a citation does and does not tell you |
| 12 | How to structure the training data |
| 13 | The discovery loop |

Figure 8 shows illustrative shapes rather than measured results, and says so in
its caption.

## License

GPL-3.0, per the repository root.
