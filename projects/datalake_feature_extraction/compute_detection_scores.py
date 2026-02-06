#!/usr/bin/env python3
"""
compute_detection_scores.py

Compute DETECTION scores for n-grams and phrases.

This is a detection problem, not classification:
- MOE: External customer has achievement
- MOP: Internal org has achievement
- NEITHER: Most content (not relevant to performance/effectiveness)

For each n-gram/phrase, compute: "How likely does this indicate
MOE/MOP content vs neither?"

Output:
    {column}_detection_score: List of scores [0.0-1.0] per element
    Higher score = more likely to indicate MOE or MOP (detected)
    Lower score = more likely to be noise (neither)

Usage:
    python compute_detection_scores.py \
        --input data.parquet \
        --output data_with_detection.parquet \
        --tag-col Narratives_sim_top_tag \
        --ngram-col 5-grams \
        --vector-col vector_5_grams
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from tqdm import tqdm
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
        self.fc = nn.Linear(100, 2)  # Binary: detected vs not detected

    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
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
    """
    Create binary detection labels:
    1 = MOE or MOP (detected)
    0 = neither (not detected)
    """
    labels = []
    for val in df[tag_col]:
        if pd.isna(val) or val == '' or val not in ['MOE', 'MOP']:
            labels.append(0)  # Neither
        else:
            labels.append(1)  # Detected (MOE or MOP)
    return np.array(labels)


def train_detection_model(df, vectors_col, tag_col, vector_dim, max_len=100, epochs=30, device='cpu'):
    """
    Train detection CNN using ALL data.

    Uses class weighting to handle imbalance (most rows are 'neither').
    """

    # Create detection labels (1 = MOE/MOP, 0 = neither)
    labels = create_detection_labels(df, tag_col)

    # Count classes
    n_detected = np.sum(labels == 1)
    n_neither = np.sum(labels == 0)
    total = len(labels)

    print(f"  Class distribution:")
    print(f"    Detected (MOE/MOP): {n_detected} ({100*n_detected/total:.1f}%)")
    print(f"    Neither:            {n_neither} ({100*n_neither/total:.1f}%)")

    if n_detected < 10:
        raise ValueError(f"Only {n_detected} detected rows - need more labeled data")

    # Prepare vectors
    vectors_list = df[vectors_col].tolist()

    X_list = []
    for vec_seq in vectors_list:
        padded, _ = prepare_single_vector(vec_seq, max_len, vector_dim)
        X_list.append(padded)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"  Class weights: neither={class_weights[0]:.2f}, detected={class_weights[1]:.2f}")

    # Create weighted sampler for balanced batches
    sample_weights = [class_weights[label].item() for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create model and move to device
    model = DetectionCNN(vector_dim).to(device)
    X, y = X.to(device), y.to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += len(yb)

            # Track detection metrics
            tp += ((preds == 1) & (yb == 1)).sum().item()
            fp += ((preds == 1) & (yb == 0)).sum().item()
            tn += ((preds == 0) & (yb == 0)).sum().item()
            fn += ((preds == 0) & (yb == 1)).sum().item()

        if (epoch + 1) % 5 == 0:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, "
                  f"Acc={correct/total:.4f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

    return model


def compute_element_detection_scores(model, vec_seq, vector_dim, max_len, device='cpu'):
    """
    Compute detection score for each element in the sequence.

    Returns list of scores where:
    - High score (close to 1.0) = likely indicates MOE/MOP content
    - Low score (close to 0.0) = likely noise/neither
    """
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

    # Detection probability (class 1 = detected)
    detection_prob = probs[0, 1].item()

    # Get per-element contribution via gradients
    model.zero_grad()
    logits[0, 1].backward()  # Gradient w.r.t. detection class

    grads = x.grad.data.abs().squeeze(0)  # (max_len, vector_dim)
    position_importance = grads.mean(dim=1).cpu().numpy()  # (max_len,)

    scores = position_importance[:actual_len]

    # Normalize and scale by detection probability
    if scores.max() > 0:
        scores = scores / scores.max()

    # Scale by overall detection probability
    scores = scores * detection_prob

    return scores.tolist()


def compute_all_detection_scores(df, model, vectors_col, vector_dim, max_len, device='cpu'):
    """Compute detection scores for all rows."""

    all_scores = []

    for idx in tqdm(range(len(df)), desc="Computing detection scores"):
        vec_seq = df.iloc[idx][vectors_col]
        scores = compute_element_detection_scores(model, vec_seq, vector_dim, max_len, device)
        all_scores.append(scores)

    return all_scores


def analyze_detection_scores(df, tokens_col, scores_col, tag_col):
    """Analyze detection score distributions by class."""

    # Collect scores by class
    detected_scores = []
    neither_scores = []
    all_token_scores = []

    for idx in range(len(df)):
        tag = df.iloc[idx][tag_col]
        tokens = df.iloc[idx].get(tokens_col)
        scores = df.iloc[idx].get(scores_col)

        if scores is None or len(scores) == 0:
            continue

        is_detected = not pd.isna(tag) and tag in ['MOE', 'MOP']

        for i, score in enumerate(scores):
            if is_detected:
                detected_scores.append(score)
            else:
                neither_scores.append(score)

            if tokens and i < len(tokens):
                token = tokens[i]
                if isinstance(token, tuple):
                    token = ' '.join(token)
                all_token_scores.append((token, score, 'detected' if is_detected else 'neither'))

    print(f"\n{'='*60}")
    print("DETECTION SCORE ANALYSIS")
    print(f"{'='*60}")

    if detected_scores:
        detected_scores = np.array(detected_scores)
        print(f"\nDetected (MOE/MOP) elements: {len(detected_scores)}")
        print(f"  Mean: {detected_scores.mean():.4f}")
        print(f"  Std:  {detected_scores.std():.4f}")
        print(f"  P50:  {np.percentile(detected_scores, 50):.4f}")
        print(f"  P90:  {np.percentile(detected_scores, 90):.4f}")
        print(f"  P95:  {np.percentile(detected_scores, 95):.4f}")

    if neither_scores:
        neither_scores = np.array(neither_scores)
        print(f"\nNeither elements: {len(neither_scores)}")
        print(f"  Mean: {neither_scores.mean():.4f}")
        print(f"  Std:  {neither_scores.std():.4f}")
        print(f"  P50:  {np.percentile(neither_scores, 50):.4f}")
        print(f"  P90:  {np.percentile(neither_scores, 90):.4f}")
        print(f"  P95:  {np.percentile(neither_scores, 95):.4f}")

    if detected_scores is not None and len(detected_scores) > 0 and neither_scores is not None and len(neither_scores) > 0:
        # Suggest threshold based on separation
        detected_p25 = np.percentile(detected_scores, 25)
        neither_p95 = np.percentile(neither_scores, 95)

        suggested_threshold = (detected_p25 + neither_p95) / 2

        print(f"\n{'='*60}")
        print("THRESHOLD RECOMMENDATION")
        print(f"{'='*60}")
        print(f"  Detected P25: {detected_p25:.4f}")
        print(f"  Neither P95:  {neither_p95:.4f}")
        print(f"  Suggested threshold: {suggested_threshold:.4f}")

        # Estimate detection performance at suggested threshold
        tp = np.sum(detected_scores >= suggested_threshold)
        fn = np.sum(detected_scores < suggested_threshold)
        fp = np.sum(neither_scores >= suggested_threshold)
        tn = np.sum(neither_scores < suggested_threshold)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n  At threshold {suggested_threshold:.4f}:")
        print(f"    Precision: {precision:.3f}")
        print(f"    Recall:    {recall:.3f}")
        print(f"    F1:        {f1:.3f}")

    # Top tokens from detected content
    if all_token_scores:
        detected_tokens = [(t, s) for t, s, c in all_token_scores if c == 'detected']
        detected_tokens.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop 20 highest-scoring tokens from detected content:")
        seen = set()
        count = 0
        for token, score in detected_tokens:
            if token not in seen:
                print(f"  {score:.4f}: {token[:60]}")
                seen.add(token)
                count += 1
                if count >= 20:
                    break


def main():
    parser = argparse.ArgumentParser(
        description='Compute detection scores (MOE/MOP vs neither)'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet file')
    parser.add_argument('--output', '-o', required=True, help='Output parquet file')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column')
    parser.add_argument('--ngram-col', required=True, help='N-gram/phrase column')
    parser.add_argument('--vector-col', required=True, help='Vector column')
    parser.add_argument('--output-col', default=None, help='Output column name')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--max-len', type=int, default=100, help='Max sequence length')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if args.output_col is None:
        args.output_col = f"{args.ngram_col}_detection_score"

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows (using ALL data for detection)")

    # Verify columns
    if args.ngram_col not in df.columns:
        raise ValueError(f"Column '{args.ngram_col}' not found")
    if args.vector_col not in df.columns:
        raise ValueError(f"Column '{args.vector_col}' not found")

    # Get vector dimension
    vector_dim = get_vector_dim(df, args.vector_col)
    print(f"Vector dimension: {vector_dim}")

    # Train detection model on ALL data
    print(f"\nTraining detection model...")
    model = train_detection_model(
        df, args.vector_col, args.tag_col, vector_dim,
        max_len=args.max_len, epochs=args.epochs, device=device
    )

    # Compute detection scores for all rows
    print(f"\nComputing detection scores...")
    scores = compute_all_detection_scores(df, model, args.vector_col, vector_dim, args.max_len, device)

    # Add to dataframe
    df[args.output_col] = scores

    # Analyze
    analyze_detection_scores(df, args.ngram_col, args.output_col, args.tag_col)

    # Save
    print(f"\nSaving to: {args.output}")
    df.to_parquet(args.output, index=False)

    print(f"\nDone! New column: {args.output_col}")


if __name__ == '__main__':
    main()
