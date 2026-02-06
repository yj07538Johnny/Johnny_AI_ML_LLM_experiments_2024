#!/usr/bin/env python3
"""
compute_predictability_scores.py

Computes per-element predictability scores for n-grams and phrases.
For each row, outputs a list of probabilities aligned with the input lists.

Example:
    Input row:
        5-grams: ["ngram A", "ngram B", "ngram C"]
        vector_5_grams: [[vec_A], [vec_B], [vec_C]]

    Output row (new column):
        5-grams_predictability: [0.85, 0.42, 0.91]

Each score represents how predictive that specific n-gram/phrase is for MOE/MOP.

Usage:
    python compute_predictability_scores.py \
        --input data.parquet \
        --output data_with_scores.parquet \
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
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tqdm import tqdm
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
        # x shape: (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv(x))
        x = F.adaptive_max_pool1d(x, 1).squeeze(2)
        return self.fc(x)

    def get_conv_features(self, x):
        """Get per-position conv features before pooling."""
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv(x))  # (batch, 100, seq_len)
        return x.permute(0, 2, 1)  # (batch, seq_len, 100)


def prepare_single_vector(vec_seq, max_len, vector_dim):
    """Prepare a single vector sequence with padding."""
    if vec_seq is None or (isinstance(vec_seq, (list, np.ndarray)) and len(vec_seq) == 0):
        return np.zeros((max_len, vector_dim), dtype=np.float32), 0

    vec_seq = np.array(vec_seq, dtype=np.float32)

    # Handle 1D vectors (summed) vs 2D sequences
    if vec_seq.ndim == 1:
        vec_seq = vec_seq.reshape(1, -1)

    actual_len = len(vec_seq)

    if len(vec_seq) < max_len:
        padding = np.zeros((max_len - len(vec_seq), vector_dim), dtype=np.float32)
        vec_seq = np.vstack([vec_seq, padding])
    else:
        vec_seq = vec_seq[:max_len]

    return vec_seq, min(actual_len, max_len)


def train_model(df, tokens_col, vectors_col, tag_col, max_len=100, epochs=20, device='cpu'):
    """Train CNN model on labeled data."""

    # Filter to labeled rows
    mask = df[tag_col].notna() & (df[tag_col] != '') & df[tag_col].isin(['MOE', 'MOP'])
    train_df = df[mask].copy()

    if len(train_df) < 10:
        raise ValueError(f"Only {len(train_df)} labeled rows - need more data")

    print(f"Training on {len(train_df)} labeled rows")

    # Get vectors and determine dimension
    vectors_list = train_df[vectors_col].tolist()

    vector_dim = 100
    for v in vectors_list:
        if v is not None and len(v) > 0:
            if isinstance(v[0], (list, np.ndarray)):
                vector_dim = len(v[0])
            else:
                vector_dim = len(v)
            break

    print(f"Vector dimension: {vector_dim}")

    # Prepare training data
    X_list = []
    for vec_seq in vectors_list:
        padded, _ = prepare_single_vector(vec_seq, max_len, vector_dim)
        X_list.append(padded)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(train_df[tag_col].values)
    y = torch.tensor(labels, dtype=torch.long)

    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")

    # Train
    model = SimpleCNN(vector_dim, num_classes).to(device)
    X, y = X.to(device), y.to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == yb).sum().item()
            total += len(yb)

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/len(dataloader):.4f}, Acc={correct/total:.4f}")

    return model, le, vector_dim, max_len


def compute_element_scores(model, vec_seq, actual_len, vector_dim, max_len, device='cpu'):
    """
    Compute predictability score for each element in the sequence.

    Uses gradient-based importance: how much does each position contribute
    to the predicted class?

    Returns list of scores, one per element in original sequence.
    """
    if actual_len == 0:
        return []

    # Prepare input
    padded, _ = prepare_single_vector(vec_seq, max_len, vector_dim)
    x = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # Forward pass
    model.eval()
    logits = model(x)
    pred_class = logits.argmax(dim=1).item()
    probs = F.softmax(logits, dim=1)
    pred_prob = probs[0, pred_class].item()

    # Backward pass to get gradients
    model.zero_grad()
    logits[0, pred_class].backward()

    # Get gradient magnitude per position
    grads = x.grad.data.abs().squeeze(0)  # (max_len, vector_dim)
    position_importance = grads.mean(dim=1).cpu().numpy()  # (max_len,)

    # Normalize to [0, 1] using softmax-style normalization
    # Higher gradient = more important for prediction
    scores = position_importance[:actual_len]

    # Scale by prediction probability to get "predictability"
    # High importance + high confidence = high predictability
    if scores.max() > 0:
        scores = scores / scores.max()  # Normalize to [0, 1]

    # Multiply by prediction confidence
    scores = scores * pred_prob

    return scores.tolist()


def compute_all_scores(df, model, vectors_col, vector_dim, max_len, device='cpu'):
    """Compute predictability scores for all rows."""

    all_scores = []

    for idx in tqdm(range(len(df)), desc="Computing scores"):
        vec_seq = df.iloc[idx][vectors_col]

        if vec_seq is None or (isinstance(vec_seq, (list, np.ndarray)) and len(vec_seq) == 0):
            all_scores.append([])
            continue

        vec_seq = np.array(vec_seq, dtype=np.float32)
        if vec_seq.ndim == 1:
            vec_seq = vec_seq.reshape(1, -1)

        actual_len = len(vec_seq)

        scores = compute_element_scores(
            model, vec_seq, actual_len, vector_dim, max_len, device
        )
        all_scores.append(scores)

    return all_scores


def analyze_scores(df, tokens_col, scores_col):
    """Analyze the distribution of predictability scores."""

    all_scores = []
    all_tokens_scores = []

    for idx in range(len(df)):
        tokens = df.iloc[idx][tokens_col]
        scores = df.iloc[idx][scores_col]

        if tokens is None or scores is None:
            continue

        if isinstance(tokens, str):
            tokens = [tokens]

        for i, score in enumerate(scores):
            all_scores.append(score)
            if i < len(tokens):
                token = tokens[i]
                if isinstance(token, tuple):
                    token = ' '.join(token)
                all_tokens_scores.append((token, score))

    if not all_scores:
        return

    all_scores = np.array(all_scores)

    print(f"\n{'='*60}")
    print(f"SCORE DISTRIBUTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total elements scored: {len(all_scores)}")
    print(f"Score range: [{all_scores.min():.4f}, {all_scores.max():.4f}]")
    print(f"Mean: {all_scores.mean():.4f}")
    print(f"Std: {all_scores.std():.4f}")
    print(f"Percentiles:")
    for p in [50, 75, 90, 95, 99]:
        val = np.percentile(all_scores, p)
        count = np.sum(all_scores >= val)
        print(f"  P{p}: {val:.4f} ({count} elements above)")

    # Top predictive items
    if all_tokens_scores:
        sorted_items = sorted(all_tokens_scores, key=lambda x: x[1], reverse=True)
        print(f"\nTop 20 most predictive items:")
        seen = set()
        count = 0
        for token, score in sorted_items:
            if token not in seen:
                print(f"  {score:.4f}: {token[:60]}")
                seen.add(token)
                count += 1
                if count >= 20:
                    break


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-element predictability scores'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet file')
    parser.add_argument('--output', '-o', required=True, help='Output parquet file')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column')
    parser.add_argument('--ngram-col', required=True, help='N-gram/phrase column (e.g., 5-grams)')
    parser.add_argument('--vector-col', required=True, help='Vector column (e.g., vector_5_grams)')
    parser.add_argument('--output-col', default=None,
                       help='Output column name (default: {ngram-col}_predictability)')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--max-len', type=int, default=100, help='Max sequence length')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set output column name
    if args.output_col is None:
        args.output_col = f"{args.ngram_col}_predictability"

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")

    # Verify columns exist
    if args.ngram_col not in df.columns:
        raise ValueError(f"Column '{args.ngram_col}' not found")
    if args.vector_col not in df.columns:
        raise ValueError(f"Column '{args.vector_col}' not found")
    if args.tag_col not in df.columns:
        raise ValueError(f"Column '{args.tag_col}' not found")

    # Train model
    print(f"\nTraining model on {args.ngram_col}...")
    model, label_encoder, vector_dim, max_len = train_model(
        df, args.ngram_col, args.vector_col, args.tag_col,
        max_len=args.max_len, epochs=args.epochs, device=device
    )

    # Compute scores for all rows
    print(f"\nComputing predictability scores for all {len(df)} rows...")
    scores = compute_all_scores(df, model, args.vector_col, vector_dim, max_len, device)

    # Add to dataframe
    df[args.output_col] = scores

    # Analyze
    analyze_scores(df, args.ngram_col, args.output_col)

    # Save
    print(f"\nSaving to: {args.output}")
    df.to_parquet(args.output, index=False)

    print(f"\nDone! New column added: {args.output_col}")
    print(f"Each cell contains a list of predictability scores aligned with {args.ngram_col}")


if __name__ == '__main__':
    main()
