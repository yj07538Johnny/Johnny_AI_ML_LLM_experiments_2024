# WAR Report Toolkit

Modular library and processing pipeline for extracting features from Weekly Activity Reports (WAR), training CNN classifiers for MOE/MOP prediction, and managing data persistence with DuckDB/Parquet.

## Migration from Notebook

This toolkit refactors the monolithic `cnn_function_experiment_2` Jupyter notebook into a clean, modular architecture:

| Notebook (before) | Toolkit (after) |
|---|---|
| Inline phrase extractors (9 functions, ~70 lines each) | `lib/phrase_extraction.py` — single `_extract_phrases()` with grammar registry |
| Duplicated CNN functions (lines 113-189 and 681-749) | `lib/cnn_classifier.py` — single set of functions |
| Repeated n-gram extraction (lines 412-447) | `lib/ngram_extraction.py` — loop-driven |
| Repeated vectorization blocks (lines 513-651) | `lib/vectorization.py` — config-driven with exclusion support |
| Pickle-based persistence | `lib/data_store.py` — DuckDB/Parquet with incremental processing |
| No exclusion list | `lib/exclusion_list.py` — managed domain stopwords |
| No symbolic extraction | `lib/symbolic_extraction.py` — NER, POS, keywords, AMR, FOL, ontology |

## Project Structure

```
war_report_toolkit/
├── lib/                          # Library modules (import these)
│   ├── __init__.py               # Public API exports
│   ├── phrase_extraction.py      # NLTK RegexpParser phrase extractors
│   ├── ngram_extraction.py       # Sentence-boundary n-gram extraction
│   ├── vectorization.py          # Word2Vec vectorization with exclusions
│   ├── cnn_classifier.py         # 1D CNN training and feature importance
│   ├── symbolic_extraction.py    # AMR, FOL, POS, NER, keywords, ontology
│   ├── exclusion_list.py         # Domain stopword management
│   ├── data_store.py             # DuckDB/Parquet persistence layer
│   └── utils.py                  # Progress bars, helpers
├── scripts/                      # Runnable scripts
│   └── process_war_reports.py    # Main processing pipeline
└── data/                         # Parquet files (created at runtime)
    ├── war_reports.parquet       # File system / path data
    ├── narrative_features.parquet # Extracted features
    ├── processing_log.parquet    # Incremental processing tracker
    └── exclusions.json           # Persisted exclusion list
```

## Data Persistence

### Parquet as Source of Truth
- All persistent data stored as Parquet files
- DuckDB provides SQL query capability over Parquet
- DataFrames are the in-memory working format

### Workflow
```
Startup:  Parquet → DataFrame (load)
Work:     DataFrame operations (extract, vectorize, train)
Shutdown: DataFrame → Parquet (save)
```

### Incremental Processing
```python
store = WarReportStore("./data")

# Check for new WAR file directories
new_dirs = store.get_unprocessed_directories("/path/to/war_files")

# Process only new directories
for d in new_dirs:
    new_df = process_directory(d)
    store.append_reports(new_df)  # Deduplicates automatically
```

## Exclusion List

High-frequency domain terms that dominate feature importance without providing meaningful signal are managed via exclusion lists:

```python
from lib import ExclusionList

exclusions = ExclusionList()
exclusions.add("national security agency")
exclusions.add("nsa")
exclusions.add_pattern(r"classification:?\s*(secret|top secret|unclassified)")

# Two modes:
# null_vector: Replace excluded terms with zero vectors (term counted but neutralized)
# omit: Skip excluded terms entirely when building document vectors
exclusions.mode = "null_vector"

# Save/load
exclusions.save("data/exclusions.json")
exclusions = ExclusionList.load("data/exclusions.json")
```

## Usage

### Command Line
```bash
# Full pipeline: check for new files, extract, vectorize, train
python scripts/process_war_reports.py \
    --data-dir ./data \
    --war-dir /path/to/war_files \
    --model-path /path/to/word2vec_model.model

# Extract features only (no CNN training)
python scripts/process_war_reports.py \
    --data-dir ./data \
    --extract-only

# Train CNN on existing features
python scripts/process_war_reports.py \
    --data-dir ./data \
    --train-only

# Include symbolic extraction (NER, POS, keywords)
python scripts/process_war_reports.py \
    --data-dir ./data \
    --symbolic

# Custom exclusion list
python scripts/process_war_reports.py \
    --data-dir ./data \
    --exclusions-file ./data/exclusions.json

# Custom n-gram sizes
python scripts/process_war_reports.py \
    --data-dir ./data \
    --ngram-sizes 2 3 4 5 10 20
```

### Python / Jupyter
```python
import sys
sys.path.insert(0, '/path/to/war_report_toolkit')

from lib import (
    WarReportStore, ExclusionList,
    extract_all_phrases, get_ngrams,
    load_word2vec_model, get_phrase_vectors,
    process_phrases_ngrams,
)

# Initialize
store = WarReportStore("./data")
df = store.load_features()

# Extract phrases for a single text
phrases = extract_all_phrases("The mission demonstrated improved capability.")

# Query with SQL
moe_reports = store.query("""
    SELECT narrative, Narratives_sim_top_tag 
    FROM features 
    WHERE Narratives_sim_top_tag = 'MOE'
    LIMIT 10
""")
```

## Connection to Hallucination Research

The symbolic extraction module mirrors the extraction library from the hallucination datalake project:

| Hallucination Project | WAR Toolkit |
|---|---|
| `code_library/symbolic_extraction/` | `lib/symbolic_extraction.py` |
| BART-Large-MNLI classification | Informed by CNN feature importance scores |
| 102-tag taxonomy (9 categories) | Mission effects / information needs taxonomy |
| Parallel sparse matrices | Per-customer-class taxonomic fingerprints |
| Orphan span discovery | Emerging customer need detection |

The feature importance scores from CNN training can drive BART-Large-MNLI text classification by providing candidate labels derived from the mission domain rather than generic categories.

## Dependencies

```
torch
numpy
pandas
nltk
gensim
duckdb
scikit-learn
tqdm
spacy (for symbolic extraction)
keybert (for keyword extraction, optional)
amrlib (for AMR extraction, optional, GPU recommended)
```
