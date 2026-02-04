# Datalake Feature Extraction Guide

A complete guide to using CNN-based feature extraction for MOE/MOP tag classification from text embeddings.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Data Format Requirements](#data-format-requirements)
4. [Script Reference](#script-reference)
5. [Complete Workflow Example](#complete-workflow-example)
6. [Customizing for Your Data](#customizing-for-your-data)
7. [Understanding the CNN Model](#understanding-the-cnn-model)
8. [Troubleshooting](#troubleshooting)

---

## Overview

This toolkit provides scripts to:

1. **Convert** pandas DataFrames to DuckDB/Parquet format for efficient storage and querying
2. **Train** CNN models to learn associations between text embeddings and MOE/MOP tags
3. **Extract** learned features from embeddings (narrative, n-grams, phrases)
4. **Write** extracted features back to parquet files as new columns

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  pandas DataFrame │ --> │  Parquet Files   │ --> │  DuckDB Database │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  CNN Feature Extractor │
                    │  (trains on labeled    │
                    │   rows only)           │
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  New Feature Columns  │
                    │  (for ALL rows)       │
                    └──────────────────────┘
```

---

## Installation

```bash
# Navigate to the project directory
cd projects/datalake_feature_extraction

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```python
import pandas as pd
import duckdb
import torch
import gensim
print("All dependencies installed successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

---

## Data Format Requirements

### Expected DataFrame Structure

Your data should have the following types of columns:

| Column Type | Description | Example Column Name | Cell Content |
|------------|-------------|---------------------|--------------|
| **Narrative** | Full text content | `narrative`, `text`, `content` | `"The patient reported..."` |
| **MOE/MOP Tag** | Classification label | `moe_mop_tag`, `label`, `tag` | `"MOE"` or `"MOP"` or empty |
| **N-gram** | List of n-grams | `ngram_5`, `ngram_10` | `["the patient", "patient reported", ...]` |
| **N-gram Embedding** | Embeddings for n-grams | `ngram_5_embedding` | `[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]` |
| **Phrase** | List of phrases | `noun_phrases` | `["the patient", "initial diagnosis", ...]` |
| **Phrase Embedding** | Embeddings for phrases | `noun_phrases_embedding` | `[[0.1, 0.2, ...], [0.3, 0.4, ...], ...]` |

### Embedding Format

Embeddings should be stored as **lists of vectors** where each vector is a Word2Vec embedding:

```python
# Example: ngram_5_embedding cell content
[
    [0.123, -0.456, 0.789, ...],  # embedding for first n-gram
    [0.234, -0.567, 0.891, ...],  # embedding for second n-gram
    ...
]
```

- Each inner list is a Word2Vec vector (typically 100-300 dimensions)
- The outer list contains one vector per n-gram/phrase in the corresponding column

### Tag Column

- Contains a single tag value (`"MOE"`, `"MOP"`, or your tag names)
- Empty/null for unlabeled rows
- Only labeled rows are used for training; features are extracted for ALL rows

---

## Script Reference

### 1. pandas_to_duckdb_parquet.py

Converts a pandas DataFrame to Parquet format with optional DuckDB registration.

**Command Line:**
```bash
python pandas_to_duckdb_parquet.py \
    --input /path/to/data.csv \
    --output /path/to/database.duckdb \
    --table my_data \
    --parquet-path /path/to/data.parquet \
    --overwrite
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--input, -i` | Yes | Input file (csv, parquet, pkl, json, xlsx) |
| `--output, -o` | Yes | Output DuckDB database path |
| `--table, -t` | No | Table name (default: `data`) |
| `--parquet-path, -p` | No | Custom parquet output path |
| `--overwrite` | No | Overwrite existing files |

**Programmatic Usage:**
```python
from pandas_to_duckdb_parquet import create_duckdb_from_dataframe
import pandas as pd

# Load or create your DataFrame
df = pd.read_pickle('my_data.pkl')

# Convert to parquet
parquet_path = create_duckdb_from_dataframe(
    df=df,
    db_path='my_database.duckdb',
    table_name='narratives',
    parquet_path='my_data.parquet',
    overwrite=True
)
```

---

### 2. cnn_feature_extractor_narrative.py

Extracts features from narrative text embeddings.

**Command Line:**
```bash
python cnn_feature_extractor_narrative.py \
    --input data.parquet \
    --output data_with_features.parquet \
    --embedding-col narrative_embedding \
    --tag-col moe_mop_tag \
    --output-feature-col narrative_features \
    --model-save-path models/narrative_model.pt \
    --num-epochs 50 \
    --batch-size 32
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input, -i` | Yes | - | Input parquet file |
| `--output, -o` | Yes | - | Output parquet file |
| `--embedding-col` | No | `NARRATIVE_EMBEDDING_COL` | Embedding column name |
| `--tag-col` | No | `MOE_MOP_TAG_COL` | Tag column name |
| `--output-feature-col` | No | `narrative_features` | Output column name |
| `--model-save-path` | No | None | Path to save trained model |
| `--num-epochs` | No | 50 | Training epochs |
| `--batch-size` | No | 32 | Batch size |
| `--learning-rate` | No | 0.001 | Learning rate |
| `--num-filters` | No | 128 | CNN filter count |
| `--max-seq-len` | No | Auto | Max sequence length |

**Output Columns Added:**
- `{output-feature-col}` - Feature vector (256 dimensions by default)
- `{output-feature-col}_pred_label` - Predicted MOE/MOP tag
- `{output-feature-col}_pred_probs` - Prediction probabilities

---

### 3. cnn_feature_extractor_ngram.py

Extracts features from n-gram embeddings. Run once per n-gram column.

**Command Line:**
```bash
python cnn_feature_extractor_ngram.py \
    --input data.parquet \
    --output data.parquet \
    --ngram-col ngram_5 \
    --embedding-col ngram_5_embedding \
    --tag-col moe_mop_tag
```

**Arguments:**
| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input, -i` | Yes | - | Input parquet file |
| `--output, -o` | Yes | - | Output parquet file |
| `--ngram-col` | No | `NGRAM_COL` | N-gram column name |
| `--embedding-col` | No | `NGRAM_EMBEDDING_COL` | Embedding column name |
| `--tag-col` | No | `MOE_MOP_TAG_COL` | Tag column name |
| `--output-feature-col` | No | `{ngram-col}_features` | Output column name |

**Iterating Over All N-gram Columns:**
```bash
# Process n-grams 1-10
for n in 1 2 3 4 5 6 7 8 9 10; do
    python cnn_feature_extractor_ngram.py \
        --input data.parquet \
        --output data.parquet \
        --ngram-col "ngram_${n}" \
        --embedding-col "ngram_${n}_embedding" \
        --tag-col moe_mop_tag
done

# Process n-grams 10-100 (by 10s)
for n in 10 20 30 40 50 60 70 80 90 100; do
    python cnn_feature_extractor_ngram.py \
        --input data.parquet \
        --output data.parquet \
        --ngram-col "ngram_${n}" \
        --embedding-col "ngram_${n}_embedding" \
        --tag-col moe_mop_tag
done
```

---

### 4. cnn_feature_extractor_phrase.py

Extracts features from phrase embeddings (noun phrases, verb phrases, etc.).

**Command Line:**
```bash
python cnn_feature_extractor_phrase.py \
    --input data.parquet \
    --output data.parquet \
    --phrase-col noun_phrases \
    --embedding-col noun_phrases_embedding \
    --tag-col moe_mop_tag
```

**Arguments:** Same as ngram extractor, but with `--phrase-col` instead of `--ngram-col`.

---

### 5. run_all_extractors.py

Batch runner that processes all embedding columns automatically.

**Command Line (Auto-detect columns):**
```bash
python run_all_extractors.py \
    --input data.parquet \
    --output data_with_features.parquet \
    --tag-col moe_mop_tag \
    --save-models models/
```

**Command Line (With config file):**
```bash
python run_all_extractors.py \
    --input data.parquet \
    --config my_config.yaml \
    --tag-col moe_mop_tag
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--input, -i` | Yes | Input parquet file |
| `--output, -o` | No | Output file (default: overwrite input) |
| `--tag-col` | Yes | Tag column name |
| `--config, -c` | No | YAML config file with column mappings |
| `--save-models` | No | Directory to save all trained models |
| `--only-type` | No | Only process: `narrative`, `ngram`, or `phrase` |

---

## Complete Workflow Example

### Step 1: Prepare Your Data

```python
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

# Load your raw data
df = pd.read_csv('raw_narratives.csv')

# Example: df has columns ['id', 'narrative', 'moe_mop_tag']
print(df.head())
```

### Step 2: Generate Embeddings (if not already done)

```python
# Train or load Word2Vec model
# Assuming you have tokenized sentences
sentences = df['narrative'].apply(lambda x: x.lower().split()).tolist()
w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)

# Function to get embeddings for a list of tokens
def get_embeddings(tokens, model):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token].tolist())
    return embeddings if embeddings else None

# Generate narrative embeddings
df['narrative_embedding'] = df['narrative'].apply(
    lambda x: get_embeddings(x.lower().split(), w2v_model)
)

# Generate n-gram embeddings (example for 5-grams)
def get_ngrams(text, n):
    tokens = text.lower().split()
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

df['ngram_5'] = df['narrative'].apply(lambda x: get_ngrams(x, 5))
df['ngram_5_embedding'] = df['ngram_5'].apply(
    lambda ngrams: [get_embeddings(ng.split(), w2v_model) for ng in ngrams] if ngrams else None
)
```

### Step 3: Convert to Parquet

```bash
python pandas_to_duckdb_parquet.py \
    --input prepared_data.pkl \
    --output database.duckdb \
    --table narratives \
    --parquet-path data.parquet
```

### Step 4: Run Feature Extraction

**Option A: Run individually**
```bash
# Narrative features
python cnn_feature_extractor_narrative.py \
    --input data.parquet \
    --output data.parquet \
    --embedding-col narrative_embedding \
    --tag-col moe_mop_tag

# N-gram features
python cnn_feature_extractor_ngram.py \
    --input data.parquet \
    --output data.parquet \
    --ngram-col ngram_5 \
    --embedding-col ngram_5_embedding \
    --tag-col moe_mop_tag
```

**Option B: Run batch**
```bash
python run_all_extractors.py \
    --input data.parquet \
    --tag-col moe_mop_tag \
    --save-models models/
```

### Step 5: Analyze Results

```python
import pandas as pd

# Load results
df = pd.read_parquet('data.parquet')

# Check new columns
feature_cols = [c for c in df.columns if 'features' in c]
print(f"Feature columns: {feature_cols}")

# Look at predictions
print(df[['moe_mop_tag', 'narrative_features_pred_label']].head(20))

# Compare accuracy on labeled data
labeled = df[df['moe_mop_tag'].notna()]
accuracy = (labeled['moe_mop_tag'] == labeled['narrative_features_pred_label']).mean()
print(f"Accuracy on labeled data: {accuracy:.2%}")
```

---

## Customizing for Your Data

### Modifying Column Names

Create a mapping file or edit the scripts directly:

**Option 1: Command-line arguments**
```bash
python cnn_feature_extractor_ngram.py \
    --embedding-col "my_custom_ngram_5_emb" \
    --ngram-col "my_custom_ngram_5" \
    --tag-col "my_label_column"
```

**Option 2: Create a shell script**
```bash
#!/bin/bash
# run_my_extraction.sh

INPUT="my_data.parquet"
OUTPUT="my_data.parquet"
TAG_COL="classification_label"

# My column naming convention: {type}_{n}_vectors
for n in 1 2 3 4 5; do
    python cnn_feature_extractor_ngram.py \
        --input $INPUT \
        --output $OUTPUT \
        --ngram-col "words_${n}" \
        --embedding-col "words_${n}_vectors" \
        --tag-col $TAG_COL
done
```

**Option 3: Config file (config.yaml)**
```yaml
columns:
  my_narrative_vectors:
    type: narrative
    source_col: my_narrative_text

  words_5_vectors:
    type: ngram
    source_col: words_5

  noun_phrase_vectors:
    type: phrase
    source_col: noun_phrases

tag_column: classification_label
```

### Adjusting Model Parameters

For different data sizes or embedding dimensions:

```bash
# Smaller dataset - reduce complexity
python cnn_feature_extractor_narrative.py \
    --num-filters 64 \
    --num-epochs 30 \
    --batch-size 16 \
    ...

# Larger dataset - increase capacity
python cnn_feature_extractor_narrative.py \
    --num-filters 256 \
    --num-epochs 100 \
    --batch-size 64 \
    ...
```

### Adding New Phrase Types

The phrase extractor works with any column containing lists of text with corresponding embeddings:

```bash
# Medical terms
python cnn_feature_extractor_phrase.py \
    --phrase-col medical_terms \
    --embedding-col medical_terms_embedding \
    ...

# Custom entities
python cnn_feature_extractor_phrase.py \
    --phrase-col extracted_entities \
    --embedding-col extracted_entities_embedding \
    ...
```

---

## Understanding the CNN Model

### Architecture

```
Input: (batch, 1, seq_len, embed_dim)
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
┌────────┐    ┌────────┐     ┌────────┐
│Conv k=3│    │Conv k=4│     │Conv k=5│   <- Parallel convolutions
│128 filt│    │128 filt│     │128 filt│
└────────┘    └────────┘     └────────┘
    │               │               │
    ▼               ▼               ▼
┌────────┐    ┌────────┐     ┌────────┐
│MaxPool │    │MaxPool │     │MaxPool │   <- Max pooling over time
└────────┘    └────────┘     └────────┘
    │               │               │
    └───────────────┼───────────────┘
                    │
                    ▼ Concatenate (384 features)
                    │
                    ▼
              ┌──────────┐
              │ FC (256) │   <- Feature extraction layer
              │ + ReLU   │
              │ + Dropout│
              └──────────┘
                    │
                    ▼
              ┌──────────┐
              │FC (n_cls)│   <- Classification layer
              └──────────┘
                    │
                    ▼
              Predictions
```

### Why This Architecture?

1. **Multiple kernel sizes (3, 4, 5)**: Capture patterns at different scales
2. **Max pooling**: Extract the most important signal regardless of position
3. **Parallel convolutions**: Learn complementary features
4. **256-dim feature layer**: Rich representation for downstream use

### Extracted Features

The 256-dimensional feature vector from the penultimate layer captures:
- Patterns associated with MOE tags
- Patterns associated with MOP tags
- Discriminative information learned from labeled data

These features can be used for:
- Further analysis
- Clustering
- Visualization (t-SNE, UMAP)
- Input to other models

---

## Troubleshooting

### Common Issues

**1. "Embedding column not found"**
```
ValueError: Embedding column 'NGRAM_EMBEDDING_COL' not found
```
**Solution:** Specify your actual column name with `--embedding-col your_column_name`

**2. "No labeled data found"**
```
ValueError: No labeled data found in column 'moe_mop_tag'
```
**Solution:** Check that your tag column has non-empty values for some rows

**3. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size: `--batch-size 16` or `--batch-size 8`

**4. Empty embeddings**
```
ValueError: Could not determine embedding dimension
```
**Solution:** Ensure embedding columns contain actual vectors, not empty lists

### Debugging Tips

```python
import pandas as pd

# Check your data
df = pd.read_parquet('data.parquet')

# List all columns
print(df.columns.tolist())

# Check embedding column content
print(df['your_embedding_col'].iloc[0])  # Should be list of vectors

# Check tag distribution
print(df['your_tag_col'].value_counts(dropna=False))

# Check for empty embeddings
empty_count = df['your_embedding_col'].apply(lambda x: x is None or len(x) == 0).sum()
print(f"Rows with empty embeddings: {empty_count}")
```

---

## Next Steps

1. **Experiment with hyperparameters**: Try different filter sizes, learning rates, epochs
2. **Analyze features**: Use t-SNE/UMAP to visualize the extracted features
3. **Compare n-gram sizes**: Which n-gram length gives best classification accuracy?
4. **Combine features**: Concatenate features from multiple extractors for ensemble analysis
5. **Export for analysis**: Query results with DuckDB for SQL-based analysis

```python
import duckdb

con = duckdb.connect('database.duckdb')
results = con.execute("""
    SELECT
        moe_mop_tag,
        narrative_features_pred_label,
        COUNT(*) as count
    FROM narratives
    WHERE moe_mop_tag IS NOT NULL
    GROUP BY 1, 2
""").df()
print(results)
```
