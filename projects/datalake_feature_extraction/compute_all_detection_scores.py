#!/usr/bin/env python3
"""
compute_all_detection_scores.py

Batch compute DETECTION scores for all n-gram and phrase columns.

Detection = Does this content indicate MOE or MOP (vs neither)?

Most content is "neither" (not relevant to performance/effectiveness).
This trains on ALL data to find the signal in the noise.

Output columns:
    {column}_detection_score: List aligned with original, scores [0.0-1.0]
    Higher = more likely MOE/MOP content
    Lower = more likely noise/neither

Usage:
    python compute_all_detection_scores.py \
        --input data.parquet \
        --output data_with_detection.parquet \
        --tag-col Narratives_sim_top_tag
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


class DetectionCNN(nn.Module):
    """CNN for detecting MOE/MOP content vs neither."""

    def __init__(self, embedding_dim, dropout=0.3):
        super(DetectionCNN, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=embedding_dim,
            out_channels=100,
            kernel_size=3,
            padding=1
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(100, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        x = self.dropout(x)
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


def create_detection_labels(df, tag_col):
    """Create binary detection labels: 1=MOE/MOP, 0=neither."""
    labels = []
    for val in df[tag_col]:
        if pd.isna(val) or val == '' or val not in ['MOE', 'MOP']:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


def train_detection_model(df, vectors_col, tag_col, vector_dim, max_len=100, epochs=20, device='cpu'):
    """Train detection CNN using ALL data with class weighting."""

    labels = create_detection_labels(df, tag_col)

    n_detected = np.sum(labels == 1)
    n_neither = np.sum(labels == 0)

    if n_detected < 5:
        return None

    vectors_list = df[vectors_col].tolist()

    X_list = []
    for vec_seq in vectors_list:
        padded, _ = prepare_single_vector(vec_seq, max_len, vector_dim)
        X_list.append(padded)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Class weights for imbalance
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Weighted sampler
    sample_weights = [class_weights[label].item() for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    model = DetectionCNN(vector_dim).to(device)
    X, y = X.to(device), y.to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for epoch in range(epochs):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def compute_element_detection_scores(model, vec_seq, vector_dim, max_len, device='cpu'):
    """Compute detection score for each element."""
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
    probs = F.softmax(logits, dim=1)
    detection_prob = probs[0, 1].item()

    model.zero_grad()
    logits[0, 1].backward()

    grads = x.grad.data.abs().squeeze(0)
    position_importance = grads.mean(dim=1).cpu().numpy()

    scores = position_importance[:actual_len]

    if scores.max() > 0:
        scores = scores / scores.max()

    scores = scores * detection_prob

    return scores.tolist()


def process_column(df, tokens_col, vectors_col, tag_col, max_len=100, epochs=20, device='cpu'):
    """Process a single column pair."""

    vector_dim = get_vector_dim(df, vectors_col)

    model = train_detection_model(df, vectors_col, tag_col, vector_dim, max_len, epochs, device)

    if model is None:
        return None

    all_scores = []
    for idx in tqdm(range(len(df)), desc="  Scoring", leave=False):
        vec_seq = df.iloc[idx][vectors_col]
        scores = compute_element_detection_scores(model, vec_seq, vector_dim, max_len, device)
        all_scores.append(scores)

    return all_scores


def get_column_pairs(df):
    """Auto-detect n-gram and phrase column pairs."""
    pairs = []

    phrase_types = [
        'noun_phrases', 'verb_phrases', 'adjective_phrases', 'adverb_phrases',
        'prepositional_phrases', 'gerund_phrases', 'infinitive_phrases',
        'participle_phrases', 'appositive_phrases'
    ]
    for pt in phrase_types:
        vec_col = f'vector_{pt}'
        if pt in df.columns and vec_col in df.columns:
            pairs.append((pt, vec_col))

    ngram_patterns = ['bigrams'] + [f'{n}-grams' for n in range(2, 11)]
    ngram_patterns += [f'{n}-grams' for n in range(20, 101, 10)]

    for ng in ngram_patterns:
        vec_col = f'vector_{ng}'.replace('-', '_')
        if ng in df.columns and vec_col in df.columns:
            pairs.append((ng, vec_col))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Compute detection scores for all n-gram and phrase columns'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet file')
    parser.add_argument('--output', '-o', required=True, help='Output parquet file')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per column')
    parser.add_argument('--max-len', type=int, default=100, help='Max sequence length')
    parser.add_argument('--columns', nargs='+', default=None, help='Specific columns to process')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows (ALL data)")

    # Show class distribution
    labels = create_detection_labels(df, args.tag_col)
    n_detected = np.sum(labels == 1)
    n_neither = np.sum(labels == 0)
    print(f"\nClass distribution:")
    print(f"  Detected (MOE/MOP): {n_detected} ({100*n_detected/len(df):.1f}%)")
    print(f"  Neither:            {n_neither} ({100*n_neither/len(df):.1f}%)")

    # Get column pairs
    if args.columns:
        pairs = []
        for col in args.columns:
            vec_col = f'vector_{col}'.replace('-', '_')
            if col in df.columns and vec_col in df.columns:
                pairs.append((col, vec_col))
    else:
        pairs = get_column_pairs(df)

    print(f"\nProcessing {len(pairs)} column pairs")

    summary = []

    for tokens_col, vectors_col in pairs:
        output_col = f'{tokens_col}_detection_score'
        print(f"\nProcessing: {tokens_col}")

        try:
            scores = process_column(
                df, tokens_col, vectors_col, args.tag_col,
                max_len=args.max_len, epochs=args.epochs, device=device
            )

            if scores is not None:
                df[output_col] = scores

                # Stats by class
                detected_scores = []
                neither_scores = []

                for idx in range(len(df)):
                    row_scores = scores[idx]
                    if row_scores:
                        tag = df.iloc[idx][args.tag_col]
                        is_detected = not pd.isna(tag) and tag in ['MOE', 'MOP']
                        if is_detected:
                            detected_scores.extend(row_scores)
                        else:
                            neither_scores.extend(row_scores)

                if detected_scores and neither_scores:
                    det_mean = np.mean(detected_scores)
                    nei_mean = np.mean(neither_scores)
                    det_p90 = np.percentile(detected_scores, 90)
                    nei_p90 = np.percentile(neither_scores, 90)
                    separation = det_mean - nei_mean

                    summary.append({
                        'column': tokens_col,
                        'detected_mean': det_mean,
                        'neither_mean': nei_mean,
                        'separation': separation,
                        'detected_p90': det_p90,
                        'neither_p90': nei_p90
                    })

                    print(f"  Detected mean: {det_mean:.4f}, Neither mean: {nei_mean:.4f}, Separation: {separation:.4f}")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Save
    print(f"\nSaving to: {args.output}")
    df.to_parquet(args.output, index=False)

    # Summary
    if summary:
        print(f"\n{'='*70}")
        print("SUMMARY: Detection Score Separation by Column")
        print(f"{'='*70}")
        print(f"{'Column':<25} {'Det Mean':>10} {'Nei Mean':>10} {'Separation':>12}")
        print("-"*70)

        summary.sort(key=lambda x: x['separation'], reverse=True)
        for s in summary:
            print(f"{s['column']:<25} {s['detected_mean']:>10.4f} {s['neither_mean']:>10.4f} {s['separation']:>12.4f}")

        # Save summary
        output_dir = Path(args.output).parent
        summary_path = output_dir / 'detection_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

        # Best columns for detection
        print(f"\nBest columns for detection (highest separation):")
        for s in summary[:5]:
            print(f"  {s['column']}: separation={s['separation']:.4f}")

    new_cols = [c for c in df.columns if c.endswith('_detection_score')]
    print(f"\nNew columns added: {len(new_cols)}")


if __name__ == '__main__':
    main()
