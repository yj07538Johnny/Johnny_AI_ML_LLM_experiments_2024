#!/usr/bin/env python3
"""
compute_all_predictability_scores.py

Batch process all n-gram and phrase columns to compute predictability scores.

For each column pair (tokens, vectors), creates a new column:
    {column}_predictability: list of scores aligned with original list

Example output structure:
    Row 0:
        5-grams: ["ngram A", "ngram B", "ngram C"]
        vector_5_grams: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
        5-grams_predictability: [0.85, 0.42, 0.91]  <-- NEW

        noun_phrases: ["the patient", "severe pain"]
        vector_noun_phrases: [[...], [...]]
        noun_phrases_predictability: [0.72, 0.88]  <-- NEW

Usage:
    python compute_all_predictability_scores.py \
        --input data.parquet \
        --output data_with_predictability.parquet \
        --tag-col Narratives_sim_top_tag
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


class SimpleCNN(nn.Module):
    """Simple 1D CNN for classification."""

    def __init__(self, embedding_dim, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=100,
            kernel_size=3,
            padding=1
        )
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        return self.fc(x)


def prepare_single_vector(vec_seq, max_len, vector_dim):
    """Prepare a single vector sequence with padding."""
    if vec_seq is None or (isinstance(vec_seq, (list, np.ndarray)) and len(vec_seq) == 0):
        return np.zeros((max_len, vector_dim), dtype=np.float32), 0

    vec_seq = np.array(vec_seq, dtype=np.float32)
    if vec_seq.ndim == 1:
        vec_seq = vec_seq.reshape(1, -1)

    actual_len = len(vec_seq)

    if len(vec_seq) < max_len:
        padding = np.zeros((max_len - len(vec_seq), vector_dim), dtype=np.float32)
        vec_seq = np.vstack([vec_seq, padding])
    else:
        vec_seq = vec_seq[:max_len]

    return vec_seq, min(actual_len, max_len)


def get_vector_dim(df, vectors_col):
    """Determine vector dimension from data."""
    for v in df[vectors_col]:
        if v is not None and len(v) > 0:
            if isinstance(v[0], (list, np.ndarray)):
                return len(v[0])
            else:
                return len(v)
    return 100


def train_model(df, vectors_col, tag_col, vector_dim, max_len=100, epochs=15, device='cpu'):
    """Train CNN model on labeled data."""

    mask = df[tag_col].notna() & (df[tag_col] != '') & df[tag_col].isin(['MOE', 'MOP'])
    train_df = df[mask].copy()

    if len(train_df) < 10:
        return None, None

    vectors_list = train_df[vectors_col].tolist()

    X_list = []
    for vec_seq in vectors_list:
        padded, _ = prepare_single_vector(vec_seq, max_len, vector_dim)
        X_list.append(padded)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)

    le = LabelEncoder()
    labels = le.fit_transform(train_df[tag_col].values)
    y = torch.tensor(labels, dtype=torch.long)

    num_classes = len(le.classes_)

    model = SimpleCNN(vector_dim, num_classes).to(device)
    X, y = X.to(device), y.to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    return model, le


def compute_element_scores(model, vec_seq, vector_dim, max_len, device='cpu'):
    """Compute predictability score for each element."""
    if vec_seq is None or len(vec_seq) == 0:
        return []

    vec_seq = np.array(vec_seq, dtype=np.float32)
    if vec_seq.ndim == 1:
        vec_seq = vec_seq.reshape(1, -1)

    actual_len = len(vec_seq)

    padded, _ = prepare_single_vector(vec_seq, max_len, vector_dim)
    x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
    x.requires_grad_(True)

    model.eval()
    logits = model(x)
    pred_class = logits.argmax(dim=1).item()
    probs = F.softmax(logits, dim=1)
    pred_prob = probs[0, pred_class].item()

    model.zero_grad()
    logits[0, pred_class].backward()

    grads = x.grad.data.abs().squeeze(0)
    position_importance = grads.mean(dim=1).cpu().numpy()

    scores = position_importance[:actual_len]

    if scores.max() > 0:
        scores = scores / scores.max()

    scores = scores * pred_prob

    return scores.tolist()


def process_column(df, tokens_col, vectors_col, tag_col, max_len=100, epochs=15, device='cpu'):
    """Process a single column pair and return predictability scores."""

    print(f"  Training model...")
    vector_dim = get_vector_dim(df, vectors_col)

    model, le = train_model(df, vectors_col, tag_col, vector_dim, max_len, epochs, device)

    if model is None:
        print(f"  Skipping - insufficient labeled data")
        return None

    print(f"  Computing scores for {len(df)} rows...")
    all_scores = []

    for idx in tqdm(range(len(df)), desc="  Scoring", leave=False):
        vec_seq = df.iloc[idx][vectors_col]
        scores = compute_element_scores(model, vec_seq, vector_dim, max_len, device)
        all_scores.append(scores)

    return all_scores


def get_column_pairs(df):
    """Auto-detect n-gram and phrase column pairs."""
    pairs = []

    # Phrase columns
    phrase_types = [
        'noun_phrases', 'verb_phrases', 'adjective_phrases', 'adverb_phrases',
        'prepositional_phrases', 'gerund_phrases', 'infinitive_phrases',
        'participle_phrases', 'appositive_phrases'
    ]
    for pt in phrase_types:
        vec_col = f'vector_{pt}'
        if pt in df.columns and vec_col in df.columns:
            pairs.append((pt, vec_col))

    # N-gram columns
    ngram_patterns = ['bigrams'] + [f'{n}-grams' for n in range(2, 11)]
    ngram_patterns += [f'{n}-grams' for n in range(20, 101, 10)]

    for ng in ngram_patterns:
        vec_col = f'vector_{ng}'.replace('-', '_')
        if ng in df.columns and vec_col in df.columns:
            pairs.append((ng, vec_col))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Compute predictability scores for all n-gram and phrase columns'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet file')
    parser.add_argument('--output', '-o', required=True, help='Output parquet file')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column')
    parser.add_argument('--epochs', type=int, default=15, help='Training epochs per column')
    parser.add_argument('--max-len', type=int, default=100, help='Max sequence length')
    parser.add_argument('--columns', nargs='+', default=None,
                       help='Specific token columns to process (default: auto-detect)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Get column pairs
    if args.columns:
        pairs = []
        for col in args.columns:
            vec_col = f'vector_{col}'.replace('-', '_')
            if col in df.columns and vec_col in df.columns:
                pairs.append((col, vec_col))
    else:
        pairs = get_column_pairs(df)

    print(f"\nFound {len(pairs)} column pairs to process")

    # Process each column
    summary = []

    for tokens_col, vectors_col in pairs:
        output_col = f'{tokens_col}_predictability'
        print(f"\nProcessing: {tokens_col} -> {output_col}")

        try:
            scores = process_column(
                df, tokens_col, vectors_col, args.tag_col,
                max_len=args.max_len, epochs=args.epochs, device=device
            )

            if scores is not None:
                df[output_col] = scores

                # Compute stats
                all_scores = [s for row_scores in scores for s in row_scores if row_scores]
                if all_scores:
                    stats = {
                        'column': tokens_col,
                        'total_elements': len(all_scores),
                        'mean': np.mean(all_scores),
                        'std': np.std(all_scores),
                        'p90': np.percentile(all_scores, 90),
                        'p95': np.percentile(all_scores, 95),
                        'p99': np.percentile(all_scores, 99),
                    }
                    summary.append(stats)
                    print(f"  Stats: mean={stats['mean']:.4f}, p90={stats['p90']:.4f}, p95={stats['p95']:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save results
    print(f"\nSaving to: {args.output}")
    df.to_parquet(args.output, index=False)

    # Print summary
    if summary:
        print(f"\n{'='*70}")
        print("SUMMARY: Predictability Score Statistics")
        print(f"{'='*70}")
        print(f"{'Column':<25} {'Count':>10} {'Mean':>8} {'P90':>8} {'P95':>8} {'P99':>8}")
        print("-"*70)
        for s in summary:
            print(f"{s['column']:<25} {s['total_elements']:>10} {s['mean']:>8.4f} "
                  f"{s['p90']:>8.4f} {s['p95']:>8.4f} {s['p99']:>8.4f}")

        # Save summary
        output_dir = Path(args.output).parent
        summary_path = output_dir / 'predictability_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

    new_cols = [c for c in df.columns if c.endswith('_predictability')]
    print(f"\nNew columns added ({len(new_cols)}):")
    for col in new_cols:
        print(f"  - {col}")


if __name__ == '__main__':
    main()
