#!/usr/bin/env python3
"""
extract_predictive_features.py

Extract predictive MOE/MOP features from prepared training data.

Input:
    filtered_df.parquet - DataFrame with:
        - N-gram columns (2-grams through 100-grams)
        - Phrase columns (noun_phrases, verb_phrases, etc.)
        - Vector columns (vector_2_grams, vector_noun_phrases, etc.)
        - Tag column (Narratives_sim_top_tag with MOE/MOP/neither labels)

Output:
    features_df.parquet - DataFrame containing:
        - Discriminative n-grams and phrases
        - Detection scores
        - Class associations (MOE vs MOP)
        - Predictive statistics

    results_df.parquet - Original data with detection scores added

Usage:
    python extract_predictive_features.py \
        --input filtered_df.parquet \
        --features-output features_df.parquet \
        --results-output results_df.parquet \
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
from collections import defaultdict
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


def prepare_vector(vec_seq, max_len, vector_dim):
    """Prepare vector sequence with padding."""
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


def create_labels(df, tag_col):
    """Create binary detection labels: 1=MOE/MOP, 0=neither."""
    labels = []
    for val in df[tag_col]:
        if pd.isna(val) or val == '' or val not in ['MOE', 'MOP']:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(labels)


def train_model(df, vectors_col, tag_col, vector_dim, max_len=100, epochs=20, device='cpu'):
    """Train detection CNN on all data with class weighting."""

    labels = create_labels(df, tag_col)
    n_detected = np.sum(labels == 1)

    if n_detected < 5:
        return None

    vectors_list = df[vectors_col].tolist()
    X_list = [prepare_vector(v, max_len, vector_dim)[0] for v in vectors_list]

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    sample_weights = [class_weights[label].item() for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    model = DetectionCNN(vector_dim).to(device)
    X, y = X.to(device), y.to(device)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    model.train()
    for _ in range(epochs):
        for xb, yb in dataloader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

    return model


def compute_scores(model, vec_seq, vector_dim, max_len, device='cpu'):
    """Compute detection score for each element in sequence."""
    if vec_seq is None or len(vec_seq) == 0:
        return []

    vec_seq = np.array(vec_seq, dtype=np.float32)
    if vec_seq.ndim == 1:
        vec_seq = vec_seq.reshape(1, -1)

    actual_len = len(vec_seq)
    padded, _ = prepare_vector(vec_seq, max_len, vector_dim)
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


def get_column_pairs(df):
    """Auto-detect token/vector column pairs."""
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
            pairs.append((pt, vec_col, 'phrase'))

    # N-gram columns
    ngram_patterns = ['bigrams'] + [f'{n}-grams' for n in range(2, 11)]
    ngram_patterns += [f'{n}-grams' for n in range(20, 101, 10)]

    for ng in ngram_patterns:
        vec_col = f'vector_{ng}'.replace('-', '_')
        if ng in df.columns and vec_col in df.columns:
            pairs.append((ng, vec_col, 'ngram'))

    return pairs


def extract_predictive_features(df, tag_col, threshold=0.5):
    """
    Extract all features above threshold into features_df.

    Returns DataFrame with columns:
        - feature_text: The n-gram or phrase text
        - feature_type: 'ngram' or 'phrase'
        - source_col: Original column name
        - detection_score: Score from model
        - tag: Associated MOE/MOP tag
        - row_idx: Source row index
    """
    features = []

    score_cols = [c for c in df.columns if c.endswith('_detection_score')]

    for score_col in score_cols:
        # Derive token column name
        token_col = score_col.replace('_detection_score', '')
        if token_col not in df.columns:
            continue

        feature_type = 'phrase' if 'phrase' in token_col else 'ngram'

        for idx in range(len(df)):
            tokens = df.iloc[idx].get(token_col)
            scores = df.iloc[idx].get(score_col)
            tag = df.iloc[idx].get(tag_col)

            if tokens is None or scores is None:
                continue

            # Only extract from labeled rows
            if pd.isna(tag) or tag not in ['MOE', 'MOP']:
                continue

            for i, (token, score) in enumerate(zip(tokens, scores)):
                if score >= threshold:
                    token_str = ' '.join(token) if isinstance(token, tuple) else str(token)
                    features.append({
                        'feature_text': token_str,
                        'feature_type': feature_type,
                        'source_col': token_col,
                        'detection_score': float(score),
                        'tag': tag,
                        'row_idx': idx
                    })

    return pd.DataFrame(features)


def aggregate_features(features_df):
    """
    Aggregate features to find most discriminative ones.

    Returns DataFrame with:
        - feature_text
        - feature_type
        - source_col
        - moe_count, mop_count, total_count
        - moe_ratio, mop_ratio
        - discriminative_score (how much it favors one class)
        - dominant_class
        - avg_detection_score
        - max_detection_score
    """
    if len(features_df) == 0:
        return pd.DataFrame()

    agg = features_df.groupby(['feature_text', 'feature_type', 'source_col']).agg({
        'detection_score': ['mean', 'max', 'count'],
        'tag': lambda x: x.tolist()
    }).reset_index()

    agg.columns = ['feature_text', 'feature_type', 'source_col',
                   'avg_score', 'max_score', 'total_count', 'tags']

    # Compute class counts
    agg['moe_count'] = agg['tags'].apply(lambda t: t.count('MOE'))
    agg['mop_count'] = agg['tags'].apply(lambda t: t.count('MOP'))
    agg['moe_ratio'] = agg['moe_count'] / agg['total_count']
    agg['mop_ratio'] = agg['mop_count'] / agg['total_count']
    agg['discriminative_score'] = abs(agg['moe_ratio'] - agg['mop_ratio'])
    agg['dominant_class'] = agg.apply(
        lambda r: 'MOE' if r['moe_count'] > r['mop_count'] else 'MOP', axis=1
    )

    # Clean up
    agg = agg.drop(columns=['tags'])
    agg = agg.sort_values('discriminative_score', ascending=False)

    return agg


def main():
    parser = argparse.ArgumentParser(
        description='Extract predictive MOE/MOP features from training data'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input parquet file (filtered_df with features/vectors)')
    parser.add_argument('--features-output', '-f', required=True,
                       help='Output parquet for features_df')
    parser.add_argument('--results-output', '-r', default=None,
                       help='Output parquet for results_df (with detection scores)')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag',
                       help='Tag column name')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Detection score threshold for feature extraction')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Training epochs per column')
    parser.add_argument('--max-len', type=int, default=100,
                       help='Max sequence length')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # =========================================================================
    # STEP 1: Load filtered_df
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: LOADING FILTERED_DF")
    print(f"{'='*70}")

    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    labels = create_labels(df, args.tag_col)
    n_moe_mop = np.sum(labels == 1)
    n_neither = np.sum(labels == 0)
    print(f"\nClass distribution:")
    print(f"  MOE/MOP (detected): {n_moe_mop} ({100*n_moe_mop/len(df):.1f}%)")
    print(f"  Neither:            {n_neither} ({100*n_neither/len(df):.1f}%)")

    # =========================================================================
    # STEP 2: Train detection models and compute scores
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: COMPUTING DETECTION SCORES")
    print(f"{'='*70}")

    pairs = get_column_pairs(df)
    print(f"Found {len(pairs)} column pairs")

    column_stats = []

    for tokens_col, vectors_col, col_type in pairs:
        score_col = f'{tokens_col}_detection_score'
        print(f"\n{tokens_col} ({col_type})...")

        try:
            vector_dim = get_vector_dim(df, vectors_col)
            model = train_model(df, vectors_col, args.tag_col, vector_dim,
                               args.max_len, args.epochs, device)

            if model is None:
                print(f"  Skipped - insufficient labeled data")
                continue

            # Compute scores for all rows
            all_scores = []
            for idx in tqdm(range(len(df)), desc="  Scoring", leave=False):
                vec_seq = df.iloc[idx][vectors_col]
                scores = compute_scores(model, vec_seq, vector_dim, args.max_len, device)
                all_scores.append(scores)

            df[score_col] = all_scores

            # Compute separation statistics
            detected_scores = []
            neither_scores = []
            for idx in range(len(df)):
                row_scores = all_scores[idx]
                if row_scores:
                    tag = df.iloc[idx][args.tag_col]
                    if not pd.isna(tag) and tag in ['MOE', 'MOP']:
                        detected_scores.extend(row_scores)
                    else:
                        neither_scores.extend(row_scores)

            if detected_scores and neither_scores:
                det_mean = np.mean(detected_scores)
                nei_mean = np.mean(neither_scores)
                separation = det_mean - nei_mean
                print(f"  Detection mean: {det_mean:.4f}, Neither mean: {nei_mean:.4f}, "
                      f"Separation: {separation:.4f}")

                column_stats.append({
                    'column': tokens_col,
                    'type': col_type,
                    'detected_mean': det_mean,
                    'neither_mean': nei_mean,
                    'separation': separation
                })

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # =========================================================================
    # STEP 3: Extract predictive features
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: EXTRACTING PREDICTIVE FEATURES")
    print(f"{'='*70}")

    print(f"Threshold: {args.threshold}")
    raw_features = extract_predictive_features(df, args.tag_col, args.threshold)
    print(f"Raw feature instances: {len(raw_features)}")

    # Aggregate to unique features
    features_df = aggregate_features(raw_features)
    print(f"Unique features: {len(features_df)}")

    # =========================================================================
    # STEP 4: Save outputs
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: SAVING OUTPUTS")
    print(f"{'='*70}")

    # Save features_df
    features_path = Path(args.features_output)
    features_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(features_path, index=False)
    print(f"Features saved to: {features_path}")

    # Save results_df if requested
    if args.results_output:
        results_path = Path(args.results_output)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(results_path, index=False)
        print(f"Results saved to: {results_path}")

    # Save column stats
    stats_path = features_path.parent / 'column_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(column_stats, f, indent=2)
    print(f"Column stats saved to: {stats_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    print(f"\nFeatures by type:")
    if len(features_df) > 0:
        type_counts = features_df.groupby('feature_type').size()
        for ft, count in type_counts.items():
            print(f"  {ft}: {count}")

    print(f"\nTop 10 discriminative features:")
    for _, row in features_df.head(10).iterrows():
        print(f"  [{row['dominant_class']}] {row['feature_text'][:50]:<50} "
              f"disc={row['discriminative_score']:.3f} score={row['avg_score']:.3f}")

    print(f"\nColumns with best separation:")
    column_stats.sort(key=lambda x: x['separation'], reverse=True)
    for stat in column_stats[:5]:
        print(f"  {stat['column']}: {stat['separation']:.4f}")

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"\nfeatures_df contains {len(features_df)} predictive features")
    print(f"Use these features to examine customer feedback dataset")


if __name__ == '__main__':
    main()
