#!/usr/bin/env python3
"""
importance_score_analysis.py

Analyze the distribution of importance scores for n-grams and phrases
to determine appropriate thresholds for discriminating predictive features.

This script:
1. Trains CNN models on each n-gram/phrase column
2. Computes gradient-based importance scores
3. Outputs distribution statistics (min, max, mean, percentiles)
4. Generates visualizations to help select thresholds

Usage:
    python importance_score_analysis.py \
        --input data.parquet \
        --tag-col Narratives_sim_top_tag \
        --output-dir analysis_results/
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available - will skip visualizations")


class SimpleCNN(nn.Module):
    """Simple 1D CNN matching the user's original architecture."""

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
        # x shape: (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        return self.fc(x)


def combine_vectors_sum(tokens, model_wv, vector_size):
    """Sum word vectors for a list of tokens."""
    vectors = []
    for token in tokens:
        if token in model_wv:
            vectors.append(model_wv[token])
    if vectors:
        return np.sum(vectors, axis=0)
    return np.zeros(vector_size)


def prepare_data(vectors, labels, max_len=100, vector_dim=100):
    """Prepare data for training - pad/truncate vectors."""
    padded_vectors = []

    for vec_seq in vectors:
        if vec_seq is None or (isinstance(vec_seq, (list, np.ndarray)) and len(vec_seq) == 0):
            vec_seq = np.zeros((max_len, vector_dim))
        else:
            vec_seq = np.array(vec_seq, dtype=np.float32)

            # Handle 1D vectors (summed) vs 2D sequences
            if vec_seq.ndim == 1:
                # Single summed vector - expand to sequence of 1
                vec_seq = vec_seq.reshape(1, -1)

            if len(vec_seq) < max_len:
                padding = np.zeros((max_len - len(vec_seq), vector_dim))
                vec_seq = np.vstack([vec_seq, padding])
            else:
                vec_seq = vec_seq[:max_len]

        padded_vectors.append(vec_seq)

    X = torch.tensor(np.array(padded_vectors), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    return X, y


def train_model(X, y, num_classes, epochs=10, batch_size=16, device='cpu'):
    """Train CNN model."""
    embedding_dim = X.shape[2]

    model = SimpleCNN(embedding_dim, num_classes).to(device)
    X, y = X.to(device), y.to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return model


def get_importance_scores(model, X, tokens_list, device='cpu'):
    """
    Compute gradient-based importance scores for each token/n-gram.

    Returns dict mapping token -> importance_score
    """
    model.eval()
    importance_scores = defaultdict(list)

    for i in range(len(X)):
        x_input = X[i:i+1].to(device)
        x_input.requires_grad_(True)

        # Forward pass
        logits = model(x_input)
        pred_class = logits.argmax(dim=1)

        # Backward pass for predicted class
        model.zero_grad()
        logits[0, pred_class].backward()

        # Get gradient magnitude as importance
        grads = x_input.grad.data.abs().squeeze(0)  # (seq_len, embed_dim)
        token_importance = grads.mean(dim=1).cpu().numpy()  # (seq_len,)

        # Map to tokens
        tokens = tokens_list[i] if i < len(tokens_list) else []
        if tokens is None:
            tokens = []
        if isinstance(tokens, str):
            tokens = [tokens]

        for j, score in enumerate(token_importance):
            if j < len(tokens):
                token = tokens[j]
                if isinstance(token, tuple):
                    token = ' '.join(token)
                importance_scores[token].append(float(score))

    # Average scores per token
    avg_scores = {
        token: np.mean(scores)
        for token, scores in importance_scores.items()
    }

    return avg_scores


def analyze_distribution(scores, name):
    """Compute distribution statistics for importance scores."""
    values = list(scores.values())

    if not values:
        return None

    values = np.array(values)

    stats = {
        'name': name,
        'count': len(values),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
        'p90': float(np.percentile(values, 90)),
        'p95': float(np.percentile(values, 95)),
        'p99': float(np.percentile(values, 99)),
    }

    # Find natural gaps/thresholds
    # Items above p90 are in top 10%
    stats['threshold_p90'] = stats['p90']
    stats['threshold_p95'] = stats['p95']
    stats['items_above_p90'] = int(np.sum(values >= stats['p90']))
    stats['items_above_p95'] = int(np.sum(values >= stats['p95']))

    return stats


def plot_distribution(scores, name, output_dir):
    """Create histogram of importance scores."""
    if not HAS_MATPLOTLIB:
        return

    values = list(scores.values())
    if not values:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.percentile(values, 90), color='r', linestyle='--', label='P90')
    axes[0].axvline(np.percentile(values, 95), color='orange', linestyle='--', label='P95')
    axes[0].axvline(np.percentile(values, 99), color='green', linestyle='--', label='P99')
    axes[0].set_xlabel('Importance Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'{name} - Importance Score Distribution')
    axes[0].legend()

    # Top items bar chart
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:20]
    if top_items:
        labels = [item[0][:30] + '...' if len(str(item[0])) > 30 else str(item[0])
                  for item in top_items]
        values_top = [item[1] for item in top_items]

        axes[1].barh(range(len(labels)), values_top)
        axes[1].set_yticks(range(len(labels)))
        axes[1].set_yticklabels(labels, fontsize=8)
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title(f'{name} - Top 20 Predictive Items')
        axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / f'{name}_distribution.png', dpi=150)
    plt.close()


def process_column(df, tokens_col, vectors_col, tag_col, device='cpu'):
    """Process a single column and return importance scores."""

    # Filter to labeled rows only
    mask = df[tag_col].notna() & (df[tag_col] != '') & df[tag_col].isin(['MOE', 'MOP'])
    filtered_df = df[mask].copy()

    if len(filtered_df) < 10:
        print(f"  Skipping - only {len(filtered_df)} labeled rows")
        return None, None

    # Get data
    tokens_list = filtered_df[tokens_col].tolist()
    vectors_list = filtered_df[vectors_col].tolist()

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(filtered_df[tag_col].values)
    num_classes = len(le.classes_)

    # Determine vector dimension
    vector_dim = 100  # default
    for v in vectors_list:
        if v is not None and len(v) > 0:
            if isinstance(v[0], (list, np.ndarray)):
                vector_dim = len(v[0])
            else:
                vector_dim = len(v)
            break

    # Prepare data
    X, y = prepare_data(vectors_list, labels, max_len=100, vector_dim=vector_dim)

    # Train model
    model = train_model(X, y, num_classes, epochs=15, device=device)

    # Get importance scores
    scores = get_importance_scores(model, X, tokens_list, device=device)

    return scores, model


def main():
    parser = argparse.ArgumentParser(
        description='Analyze importance score distributions for threshold selection'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet file')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column name')
    parser.add_argument('--output-dir', '-o', default='analysis_results', help='Output directory')
    parser.add_argument('--columns', nargs='+', default=None,
                       help='Specific columns to analyze (default: auto-detect)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define column mappings
    if args.columns:
        # User specified columns
        column_pairs = []
        for col in args.columns:
            vector_col = f'vector_{col}' if not col.startswith('vector_') else col
            tokens_col = col.replace('vector_', '') if col.startswith('vector_') else col
            if tokens_col in df.columns and vector_col in df.columns:
                column_pairs.append((tokens_col, vector_col))
    else:
        # Auto-detect phrase and n-gram columns
        column_pairs = []

        # Phrase columns
        phrase_types = [
            'noun_phrases', 'verb_phrases', 'adjective_phrases', 'adverb_phrases',
            'prepositional_phrases', 'gerund_phrases', 'infinitive_phrases',
            'participle_phrases', 'appositive_phrases'
        ]
        for pt in phrase_types:
            if pt in df.columns and f'vector_{pt}' in df.columns:
                column_pairs.append((pt, f'vector_{pt}'))

        # N-gram columns
        ngram_names = ['bigrams'] + [f'{n}-grams' for n in range(2, 11)]
        ngram_names += [f'{n}-grams' for n in range(20, 101, 10)]

        for ng in ngram_names:
            vec_col = f'vector_{ng}'.replace('-', '_')
            if ng in df.columns and vec_col in df.columns:
                column_pairs.append((ng, vec_col))

    print(f"\nFound {len(column_pairs)} column pairs to analyze")

    # Process each column
    all_stats = []
    all_scores = {}

    for tokens_col, vectors_col in column_pairs:
        print(f"\nProcessing: {tokens_col}")

        try:
            scores, model = process_column(df, tokens_col, vectors_col, args.tag_col, device)

            if scores is None:
                continue

            # Analyze distribution
            stats = analyze_distribution(scores, tokens_col)
            if stats:
                all_stats.append(stats)
                all_scores[tokens_col] = scores

                print(f"  Count: {stats['count']}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Percentiles: P50={stats['median']:.4f}, P90={stats['p90']:.4f}, P95={stats['p95']:.4f}")

                # Plot
                plot_distribution(scores, tokens_col, output_dir)

                # Save top items
                top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:100]
                with open(output_dir / f'{tokens_col}_top100.json', 'w') as f:
                    json.dump(top_items, f, indent=2)

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save summary statistics
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(output_dir / 'distribution_summary.csv', index=False)

        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY: Importance Score Distributions")
        print("="*80)
        print(f"\n{'Column':<25} {'Min':>8} {'Max':>8} {'Mean':>8} {'P90':>8} {'P95':>8} {'P99':>8}")
        print("-"*80)
        for stats in all_stats:
            print(f"{stats['name']:<25} {stats['min']:>8.4f} {stats['max']:>8.4f} "
                  f"{stats['mean']:>8.4f} {stats['p90']:>8.4f} {stats['p95']:>8.4f} {stats['p99']:>8.4f}")

        # Recommend thresholds
        print("\n" + "="*80)
        print("RECOMMENDED THRESHOLDS")
        print("="*80)

        all_p90 = [s['p90'] for s in all_stats]
        all_p95 = [s['p95'] for s in all_stats]
        all_p99 = [s['p99'] for s in all_stats]

        print(f"\nGlobal P90 range: [{min(all_p90):.4f}, {max(all_p90):.4f}]")
        print(f"Global P95 range: [{min(all_p95):.4f}, {max(all_p95):.4f}]")
        print(f"Global P99 range: [{min(all_p99):.4f}, {max(all_p99):.4f}]")

        print(f"\nSuggested thresholds:")
        print(f"  Conservative (top 1%):  {np.mean(all_p99):.4f}")
        print(f"  Moderate (top 5%):      {np.mean(all_p95):.4f}")
        print(f"  Liberal (top 10%):      {np.mean(all_p90):.4f}")

    print(f"\nResults saved to: {output_dir}/")


if __name__ == '__main__':
    main()
