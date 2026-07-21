# Model-Management Architecture

**Design specification for implementation.** Companion document: `frozen-trunk-multihead-design.md` (referenced as **[MH]** — the first tenant of this architecture). Terminology is shared; these documents are internally consistent.

---

## 1. Purpose

Make every trained model **reproducible by construction**: every input to training — the data pull, the labels, the corrections, the parent model, the hyperparameters — is a versioned, re-executable artifact.

> **Reproducibility means the recipe re-executes — not that someone kept a copy of the dataframe.**

## 2. Storage Roles

- **Filesystem** (the training-data filesystem): all heavy artifacts — datasets, embedding caches, model weights, model cards, logs.
- **Database** (the spine): identity, versions, storage locations, parameters, and every relationship as a foreign key. Nothing is known to the platform unless it is a row; nothing heavy lives in a row.

## 3. Data Source Abstraction

Sources are typed by **role**, not technology:

| Role | In [MH] | Function |
|---|---|---|
| **A** metadata source | metadata repository | defines populations (windowed pulls), supplies join keys |
| **B** content source | content repository | payload the model consumes, joined by key |
| **C** requirements source | requirements repository | label authority; deterministic rules produce labels |

Any program with a population source, payload source, and labeling authority maps onto this schema.

## 4. Versioned Artifacts

- **Dataset recipe:** A query + time window + snapshot date + B join spec. Executing it materializes a **dataset** (storage path, record count, corpus hash registered).
- **Label rule:** per-customer C requirements reference + requirements version + rule hash. Applying it to a dataset yields a **label set** (versioned, class balance recorded).
- **Correction overlay:** versioned per-record label amendments from human adjudication [MH §5] — adjudicator, timestamp, source model version recorded. **C is never mutated**; effective labels = label set ⊕ overlay; every training job records the overlay version consumed.
- **Feature set:** versioned feature-extraction spec φ for metadata augmentation [MH §8].
- **Model:** trunk | head | baseline — weights on the filesystem, registered with model card path, dataset/label-set/overlay/feature-set versions, hyperparameters, metrics, threshold, and `parent_model_id`.
- **Manifest:** the deployment truth — active trunk + active head versions. Git-committed; its log is the deployment history.

## 5. Registry Schema

```
data_sources(id, name, role[A|B|C], connection_ref)
dataset_recipes(id, a_query, time_window, snapshot_date, b_join_spec, created_at)
datasets(id, recipe_id, storage_path, n_records, corpus_hash)
label_rules(id, customer_id, c_requirements_ref, requirements_version, rule_hash)
label_sets(id, dataset_id, label_rule_id, version, storage_path, class_balance)
correction_overlays(id, label_set_id, version, storage_path)
corrections(overlay_id, record_id, old_label, new_label, adjudicator,
            source_model_id, at)
feature_sets(id, version, extraction_spec, storage_path)
models(id, kind[trunk|head|baseline], name, version, parent_model_id,
       dataset_id, label_set_id, overlay_version, feature_set_id,
       trunk_model_id, storage_path, model_card_path,
       hyperparams_json, metrics_json, threshold, trained_at)
training_jobs(id, model_id, cadence[monthly|annual|on_demand],
              recipe_id, status, started_at, finished_at, log_path)
manifest(deployed_at, trunk_model_id, head_model_ids_json, committed_by)
```

Model cards are human-readable files (scope, intended use, data description, metrics, caveats); the row stores the location, not the content.

## 6. Model Lineage DAG

`parent_model_id` makes history a DAG: trunks fine-tuned from predecessors, heads warm-started from prior heads, every head referencing the trunk (and feature set) it consumes. The audit query for any deployed prediction resolves in one join chain: head version → dataset + label-set + overlay versions (including individual adjudications) → parent chain → trunk.

## 7. Orchestrated Cadence

All cadences are managed job chains writing rows at every step.

- **Monthly (head refresh — MVP heartbeat):** incremental recipe → append-embed new records → retrain all heads on effective labels → threshold recalibration → **regression gate** (metrics vN vs vN−1 per head) → manifest commit. Minutes of CPU + one small embedding job.
- **Annual (trunk refresh):** trunk fine-tune/re-DAPT recorded with parent lineage → full corpus re-embed → rebuild all heads → **atomic manifest cutover** (one commit).
- **On-demand:** a C requirements change or adjudication batch runs the same machinery on one head.

Rollback at any cadence = manifest revert. Artifacts are immutable; versions only accumulate.

## 8. Reproducibility Guarantees (structural, not procedural)

For any registered model:
1. Its **dataset** is reconstructable — re-execute the recipe against the recorded snapshot.
2. Its **labels** are reconstructable — label-rule version ⊕ overlay version.
3. Its **training** is re-runnable — recorded hyperparameters + parent model.
4. Its **deployment history** is the git log of the manifest.

Enforced by foreign keys and immutable artifacts, not operator discipline.

## 9. Failure Modes and Guards

| Failure | Guard |
|---|---|
| Data pull unreproducible | Recipes with snapshot dates are the only acquisition path |
| Label provenance lost | Label rules + overlays versioned; jobs record consumed versions |
| System of record corrupted by corrections | Overlays never mutate C |
| Untracked fine-tune | parent_model_id required; training only via recorded jobs |
| Silent regression | Regression gate before any manifest commit |
| Deployment ambiguity | Manifest is single truth; git log is history |

## 10. Implementation Checklist

- [ ] Schema migrations (§5)
- [ ] Recipe executor with snapshot recording + dataset registration
- [ ] Label-rule engine over C + label-set registration
- [ ] Overlay writer (adjudication API) + effective-label join
- [ ] Model registrar: artifact write, model-card template, metadata capture
- [ ] Job orchestrator: monthly chain, annual chain, on-demand single-head; regression gate
- [ ] Manifest manager: commit, assert-compatibility loader hooks, revert
- [ ] Lineage query API (prediction → full provenance chain)
- [ ] git-lfs/DVC setup for datasets, caches, trunks
