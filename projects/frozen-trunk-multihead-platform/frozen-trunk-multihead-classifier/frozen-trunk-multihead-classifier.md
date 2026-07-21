# Frozen-Trunk Multi-Head Classifier for Citation-Fidelity Remediation

**Design specification for implementation.** Companion document: `model-management-architecture.md` (the registry, recipes, versioning, and orchestration referenced throughout as **[MM]**). These two documents are internally consistent; terminology is shared.

---

## 1. Purpose and Theory of Operation

Customer products cite the requirements they satisfy. Citation is **deterministic where it occurs but incomplete** — some records that meet a requirement carry no citation, and the affected customer never receives that product. This system learns the discipline of attribution from the deterministic citation labels and applies it **predictively from text**, surfacing uncited-but-relevant records for human adjudication. The objective is **citation fidelity**: customers get their products.

The design goal is **best sustained performance under drift**, not perfection. Requirements change, citation practice varies, data grows. The architecture makes the frequent corrective operations cost seconds and schedules the expensive one annually.

## 2. Architectural Contract

> **The trunk is frozen after DAPT. Its pooled embedding space is a fixed interface. Heads are pure consumers of that interface.**

Consequences: heads share no trainable state (retraining one cannot affect another); the corpus embeds once per trunk version into a cache (all head training is CPU, seconds); a head is valid only against the (trunk version, feature-set version) it was trained on — recorded in metadata, asserted at load.

## 3. Data Sources and Acquisition (A/B/C)

- **A — metadata repository.** One-year windowed pull defines the training population; supplies join IDs.
- **B — content repository.** Text content, joined to A by ID. This is what the trunk embeds.
- **C — requirements repository.** Label authority. Customer c's requirements define a deterministic labeling function `L_c` via citations in product records; applied over the year of A+B data it yields the label column for head c. **Labels are stored in the dataframe, one column per customer.**

Every pull is a **dataset recipe** (A query + window + snapshot date + B join spec) and every labeling is a **label rule** (C requirements ref + version + rule hash), both registered per [MM]. A requirements change in C = new `L_c` version = new label-set version = seconds-scale retrain of head c. **C is upstream of the deployment manifest.**

### Phased rollout
- **Phase 1 (pipeline proof, customer 1):** pull A → join B → label via C → train → infer → write `pred_customer_1` column → validate. The phase-1 model registers as the permanent comparison baseline (the 98.3% single-head model is its ancestor).
- **Phase 2 (scale):** replicate labeling for all 50 customers → 50-column label frame → train 50 heads against cached embeddings.

## 4. Label Semantics: Positive-Unlabeled (PU)

Cited = **clean positive**. Uncited = **contaminated negative** (true negatives + missed citations — the target discovery).

- **Training:** standard supervised training on the noisy labels is the baseline; text signal generalizes past citation gaps. Escalation if contamination is heavy: nnPU loss with estimated class prior, or spy-based negative cleaning. Per-head `pos_weight` for imbalance.
- **Evaluation:** score against **adjudicated** labels on sampled disagreements, not raw C labels. A "false positive" vs. C is either model error or discovered missed citation; only adjudication distinguishes them. Metrics carry their label-set + overlay version.
- **Retroactive:** re-examine the 1.7% baseline "errors" — some are plausibly correct discoveries.

## 5. HITL Adjudication Loop (iterative, continuous)

1. Head c scores the corpus; high-confidence disagreements with `L_c` enter the review queue. Customer feedback on delivered products enters the same queue.
2. Human adjudicates each item:
   - **Missed citation** → customer receives the product; a label correction is written to the versioned **correction overlay** [MM]. C is never mutated.
   - **Model error** → confirmed hard negative for retraining.
3. Overlay version bump → head c retrains against cached embeddings (seconds) → new head version → manifest bump.
4. Recurring pathologies are traced to training data via recipe + lineage records [MM] and fixed at the source.

The loop does not converge to perfect and is not intended to; it **holds** best achievable performance.

## 6. Training Pipeline

**Trunk (annual; GPU).** BART text-infilling denoising (Poisson λ=3 spans, ~30% corruption) on the unlabeled corpus; fp16; lr 1e-5–5e-5; 1–3 epochs; sequence length capped to real field length; corpus deduplicated against all frozen test splits pre-DAPT. Save `trunk/vN`; embed corpus once → row-aligned cache. T4-16GB viable with grad accumulation; prefer stable local hardware for long runs.

**Heads (monthly + on-demand; CPU, seconds).** Fit `Linear(1024→1)` (escalation: MLP 1024→256→1) with weighted BCEWithLogitsLoss against cached embeddings and effective labels (label set ⊕ overlay); sweep threshold on validation; evaluate per-class P/R/F1 on the frozen test split; write `head.pt` + `meta.json` (incl. trunk version, feature-set version, label-set + overlay versions, threshold, metrics); bump manifest. Rollback = manifest revert.

## 7. Operational Cadence

- **Monthly — MVP heartbeat.** Ingest new month of A/B; **append-embed** only new records; apply current label rules + overlays; retrain all 50 heads; recalibrate thresholds; **regression gate** (each head vs. its prior version) before manifest commit. Minutes of CPU + one small embed job.
- **Annual — trunk refresh.** Fine-tune/re-DAPT on grown corpus → `trunk/vN+1` (parent lineage recorded); full re-embed; rebuild all 50 heads; **atomic manifest cutover** (trunk + all heads, one commit). Annual review also revisits feature augmentation and PU escalation.
- **On-demand.** A C requirements change or an adjudication batch runs the same machinery on one head.

All runs are recorded, versioned training jobs per [MM]; any model is reproducible by recipe re-execution.

## 8. Feature Augmentation (if text alone is insufficient)

Concatenate structured features from A: `z' = [z ; φ(metadata)]`, heads widen to 1024+d. φ is a **versioned feature-extraction spec** with its own recipe [MM]. Head validity becomes the pair (trunk version, feature-set version), both asserted at load. Trunk untouched; contract preserved.

## 9. Registry and Deployment (summary — full spec in [MM])

Filesystem artifacts (trunk, heads, thresholds, model cards) + database spine (identity, versions, locations, lineage) + `manifest.json` as deployment truth. Session init loads trunk, asserts per-head compatibility, registers thresholds. Head artifacts in plain git; trunk + embeddings in git-lfs/DVC. `git log manifest.json` is the deployment history.

## 10. Failure Modes and Guards

| Failure | Guard |
|---|---|
| Missed citations trained as negatives | PU framing; disagreement mining; adjudication; correction overlay |
| Metrics mislead | Evaluate vs adjudicated labels; metrics versioned with label set |
| Stale trunk/feature-set head served | Load-time version assertions |
| Silent regression on monthly retrain | Per-head metric comparison vN vs vN−1 gates the manifest |
| DAPT test contamination | Pre-DAPT dedup vs all frozen test splits |
| Imbalance masked by accuracy | Per-head F1/P/R mandatory |
| Requirements drift | C-versioned label rules → label-set version → head retrain |

## 11. Implementation Checklist

- [ ] A/B/C connectors + dataset-recipe executor (recorded queries, snapshot dates)
- [ ] Deterministic labeler: `L_c` per customer from C; 50-column label frame writer
- [ ] DAPT script (text-infilling collator, fp16, checkpoint-resume)
- [ ] Corpus embedder + append-embed incremental mode
- [ ] Head trainer: weighted loss, threshold sweep, frozen-split eval, artifact + metadata writer, manifest bump
- [ ] Disagreement miner + adjudication queue UI (approve→overlay write / reject→hard negative)
- [ ] Correction-overlay storage + effective-label join (label set ⊕ overlay)
- [ ] Monthly cadence orchestration with regression gate; annual cutover job
- [ ] Feature-augmentation path: φ spec format, versioning, widened heads
- [ ] Registry integration per model-management-architecture.md
