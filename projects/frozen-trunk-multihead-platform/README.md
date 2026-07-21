# Frozen-Trunk Multi-Head Platform

Three companion design documents for a production system that turns a legacy
analytical corpus into a provenance-tracked, human-gated automated
product-delivery platform. The unifying pattern is **one adapted substrate,
many cheap consumers**: a domain-adapted transformer trunk is frozen and its
pooled embedding space becomes a fixed interface that many independently
versioned heads (and downstream jobs) consume.

## Documents

| Folder | Document | What it specifies |
|--------|----------|-------------------|
| `frozen-trunk-multihead-classifier/` | A Frozen-Trunk Multi-Head Architecture for Citation-Fidelity Remediation via Positive-Unlabeled Learning | A frozen `distilbart-mnli-12-3` trunk feeding 50 per-customer binary heads; PU learning over incomplete citation labels; a human-in-the-loop adjudication loop; monthly-head / annual-trunk cadence. |
| `model-management-architecture/` | A Recipe-Based Model-Management Architecture for Versioned, Reproducible Training | The registry/versioning spine underneath the classifier: dataset recipes, label rules, correction overlays, feature sets, model lineage DAG, and a git-committed deployment manifest. "Reproducible by construction." |
| `code-mining-platform/` | Mining Legacy Python Corpora into a Provenance-Tracked Automation Platform with Human-in-the-Loop Delivery | The delivery envelope: five layers (discovery, clone detection, library consolidation, golden-master validation, generative HITL delivery) over a single catalog database. Scheduled periodic reports run on the same machinery. |

The first two are companions (they cross-reference as **[MH]** ↔ **[MM]**); the
third is a standalone IEEE-format paper.

## Note on the trunk dimension

The trunk is `distilbart-mnli-12-3`, distilled from `bart-large-mnli`, whose
`d_model` is **1024**. The `.tex` and `.md` sources here use 1024 for the pooled
embedding `z` and the head/augmentation dimensions. (An earlier draft used 768,
the `bart-base` size; that has been corrected.)

## Building

The two design specs and the IEEE paper compile with `pdflatex` (the classifier
and code-mining papers use `IEEEtran`). All three compiled PDFs are included
alongside their `.tex` sources and were produced from the corrected sources
(the classifier PDF reflects the 1024-dimension fix).

## Author

Johnny Morgan — Information Systems, UMBC.
