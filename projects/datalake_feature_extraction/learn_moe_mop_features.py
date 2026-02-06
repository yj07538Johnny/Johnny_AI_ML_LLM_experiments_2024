#!/usr/bin/env python3
"""
learn_moe_mop_features.py

Phase 1: Learn MOE/MOP feature patterns from prepared training data.

This script reads filtered_df (with all feature data and vectors in place),
learns what distinguishes MOE and MOP content from noise, and exports
the learned patterns for application to new datasets.

Input:
    - Parquet file with filtered_df containing:
        - N-gram columns (2-grams through 100-grams)
        - Phrase columns (noun_phrases, verb_phrases, etc.)
        - Vector columns (vector_2_grams, vector_noun_phrases, etc.)
        - Tag column (Narratives_sim_top_tag with MOE/MOP labels)

Output:
    - Detection scores added to dataframe
    - Learned feature patterns exported to JSON
    - Trained models saved for application phase
    - Results dataframe saved to parquet

Usage:
    python learn_moe_mop_features.py \
        --input filtered_df.parquet \
        --output-dir learned_features/ \
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
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
import json
import pickle
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


def normalize_text(text):
    """Normalize text for comparison."""
    if text is None:
        return ""
    if isinstance(text, tuple):
        text = ' '.join(text)
    return str(text).lower().strip()


def ngram_overlaps_phrase(ngram, phrase, min_overlap=0.5):
    """Check if n-gram overlaps with phrase by word overlap."""
    ngram_words = set(normalize_text(ngram).split())
    phrase_words = set(normalize_text(phrase).split())

    if not ngram_words or not phrase_words:
        return False

    overlap = ngram_words & phrase_words
    smaller_size = min(len(ngram_words), len(phrase_words))
    overlap_ratio = len(overlap) / smaller_size if smaller_size > 0 else 0

    return overlap_ratio >= min_overlap


def extract_discriminative_patterns(df, ngram_cols, phrase_cols, tag_col, threshold=0.5):
    """Extract n-gram/phrase combinations that discriminate MOE/MOP."""
    patterns = []

    for idx in tqdm(range(len(df)), desc="Extracting patterns"):
        tag = df.iloc[idx].get(tag_col)
        if pd.isna(tag) or tag not in ['MOE', 'MOP']:
            continue

        for ngram_col in ngram_cols:
            score_col = f'{ngram_col}_detection_score'
            ngrams = df.iloc[idx].get(ngram_col)
            scores = df.iloc[idx].get(score_col)

            if ngrams is None or scores is None:
                continue

            for i, (ng, sc) in enumerate(zip(ngrams, scores)):
                if sc < threshold:
                    continue

                ng_str = ' '.join(ng) if isinstance(ng, tuple) else str(ng)

                for phrase_col in phrase_cols:
                    phrases = df.iloc[idx].get(phrase_col)
                    if phrases is None:
                        continue

                    for phrase in phrases:
                        if ngram_overlaps_phrase(ng, phrase):
                            phrase_str = ' '.join(phrase) if isinstance(phrase, tuple) else str(phrase)
                            patterns.append({
                                'ngram': ng_str,
                                'ngram_col': ngram_col,
                                'phrase': phrase_str,
                                'phrase_col': phrase_col,
                                'score': float(sc),
                                'tag': tag
                            })

    return patterns


def aggregate_patterns(patterns):
    """Aggregate patterns to find most discriminative combinations."""
    from collections import defaultdict

    pattern_counts = defaultdict(lambda: {'MOE': 0, 'MOP': 0, 'scores': []})

    for p in patterns:
        key = (p['ngram'], p['phrase_col'])
        pattern_counts[key][p['tag']] += 1
        pattern_counts[key]['scores'].append(p['score'])

    aggregated = []
    for (ngram, phrase_col), counts in pattern_counts.items():
        total = counts['MOE'] + counts['MOP']
        if total < 2:
            continue

        moe_ratio = counts['MOE'] / total
        mop_ratio = counts['MOP'] / total
        discriminative_score = abs(moe_ratio - mop_ratio)

        aggregated.append({
            'ngram': ngram,
            'phrase_col': phrase_col,
            'moe_count': counts['MOE'],
            'mop_count': counts['MOP'],
            'total_count': total,
            'moe_ratio': moe_ratio,
            'mop_ratio': mop_ratio,
            'discriminative_score': discriminative_score,
            'dominant_class': 'MOE' if counts['MOE'] > counts['MOP'] else 'MOP',
            'avg_detection_score': np.mean(counts['scores'])
        })

    aggregated.sort(key=lambda x: x['discriminative_score'], reverse=True)
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description='Learn MOE/MOP feature patterns from training data'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet file with filtered_df')
    parser.add_argument('--output-dir', '-o', required=True, help='Output directory for learned features')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs per column')
    parser.add_argument('--max-len', type=int, default=100, help='Max sequence length')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection score threshold')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prepared training data
    print(f"\n{'='*70}")
    print("PHASE 1: LOADING PREPARED TRAINING DATA")
    print(f"{'='*70}")
    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Show class distribution
    labels = create_detection_labels(df, args.tag_col)
    n_detected = np.sum(labels == 1)
    n_neither = np.sum(labels == 0)
    print(f"\nClass distribution:")
    print(f"  Detected (MOE/MOP): {n_detected} ({100*n_detected/len(df):.1f}%)")
    print(f"  Neither:            {n_neither} ({100*n_neither/len(df):.1f}%)")

    # Get column pairs
    pairs = get_column_pairs(df)
    print(f"\nFound {len(pairs)} column pairs to process")

    # Phase 2: Train detection models and compute scores
    print(f"\n{'='*70}")
    print("PHASE 2: TRAINING DETECTION MODELS")
    print(f"{'='*70}")

    models = {}
    model_metadata = {}

    for tokens_col, vectors_col in pairs:
        output_col = f'{tokens_col}_detection_score'
        print(f"\nProcessing: {tokens_col}")

        try:
            vector_dim = get_vector_dim(df, vectors_col)
            model = train_detection_model(
                df, vectors_col, args.tag_col, vector_dim,
                max_len=args.max_len, epochs=args.epochs, device=device
            )

            if model is None:
                print(f"  Skipped - insufficient data")
                continue

            # Compute detection scores
            all_scores = []
            for idx in tqdm(range(len(df)), desc="  Scoring", leave=False):
                vec_seq = df.iloc[idx][vectors_col]
                scores = compute_element_detection_scores(model, vec_seq, vector_dim, args.max_len, device)
                all_scores.append(scores)

            df[output_col] = all_scores

            # Compute stats
            detected_scores = []
            neither_scores = []

            for idx in range(len(df)):
                row_scores = all_scores[idx]
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
                separation = det_mean - nei_mean
                print(f"  Detected mean: {det_mean:.4f}, Neither mean: {nei_mean:.4f}, Separation: {separation:.4f}")

                # Save model
                model_path = output_dir / f'model_{tokens_col.replace("-", "_")}.pt'
                torch.save(model.state_dict(), model_path)
                models[tokens_col] = model

                model_metadata[tokens_col] = {
                    'vector_dim': vector_dim,
                    'max_len': args.max_len,
                    'detected_mean': det_mean,
                    'neither_mean': nei_mean,
                    'separation': separation,
                    'detected_p90': np.percentile(detected_scores, 90),
                    'neither_p90': np.percentile(neither_scores, 90)
                }

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Phase 3: Extract discriminative patterns
    print(f"\n{'='*70}")
    print("PHASE 3: EXTRACTING DISCRIMINATIVE PATTERNS")
    print(f"{'='*70}")

    ngram_cols = [col for col, _ in pairs if 'gram' in col]
    phrase_cols = [col for col, _ in pairs if 'phrase' in col]

    patterns = extract_discriminative_patterns(
        df, ngram_cols, phrase_cols, args.tag_col, args.threshold
    )
    print(f"Found {len(patterns)} raw pattern instances")

    aggregated = aggregate_patterns(patterns)
    print(f"Aggregated to {len(aggregated)} unique patterns")

    # Phase 4: Determine optimal thresholds
    print(f"\n{'='*70}")
    print("PHASE 4: DETERMINING OPTIMAL THRESHOLDS")
    print(f"{'='*70}")

    thresholds = {}
    for col, meta in model_metadata.items():
        # Threshold between detected P25 and neither P90
        det_p25 = meta['detected_mean'] - 0.5 * (meta['detected_mean'] - meta['neither_mean'])
        nei_p90 = meta['neither_p90']
        suggested = (det_p25 + nei_p90) / 2
        thresholds[col] = {
            'suggested': suggested,
            'conservative': nei_p90 * 1.1,  # Higher threshold, fewer false positives
            'aggressive': suggested * 0.8   # Lower threshold, more recall
        }
        print(f"  {col}: suggested={suggested:.4f}")

    # Phase 5: Save all learned features
    print(f"\n{'='*70}")
    print("PHASE 5: SAVING LEARNED FEATURES")
    print(f"{'='*70}")

    # Save results dataframe
    results_path = output_dir / 'training_results.parquet'
    df.to_parquet(results_path, index=False)
    print(f"Results saved to: {results_path}")

    # Save model metadata
    metadata_path = output_dir / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")

    # Save discriminative patterns
    patterns_path = output_dir / 'discriminative_patterns.json'
    with open(patterns_path, 'w') as f:
        json.dump(aggregated[:500], f, indent=2)  # Top 500 patterns
    print(f"Patterns saved to: {patterns_path}")

    # Save thresholds
    thresholds_path = output_dir / 'thresholds.json'
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    print(f"Thresholds saved to: {thresholds_path}")

    # Save configuration for application phase
    config = {
        'tag_col': args.tag_col,
        'max_len': args.max_len,
        'threshold': args.threshold,
        'epochs': args.epochs,
        'column_pairs': [(t, v) for t, v in pairs],
        'ngram_cols': ngram_cols,
        'phrase_cols': phrase_cols,
        'device': device
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: LEARNED FEATURES")
    print(f"{'='*70}")
    print(f"Training rows: {len(df)}")
    print(f"Models trained: {len(models)}")
    print(f"Discriminative patterns: {len(aggregated)}")

    print(f"\nTop 10 discriminative patterns:")
    for p in aggregated[:10]:
        print(f"  {p['ngram'][:40]:<40} in {p['phrase_col']:<20} "
              f"MOE:{p['moe_count']:>3} MOP:{p['mop_count']:>3} "
              f"disc:{p['discriminative_score']:.3f}")

    print(f"\nTop columns by separation:")
    sorted_cols = sorted(model_metadata.items(), key=lambda x: x[1]['separation'], reverse=True)
    for col, meta in sorted_cols[:5]:
        print(f"  {col}: separation={meta['separation']:.4f}")

    print(f"\nOutput directory: {output_dir}")
    print("\nReady for Phase 2: Apply learned features to customer feedback dataset")
    print("Run: python apply_moe_mop_features.py --input customer_feedback.parquet "
          f"--learned-dir {output_dir} --output predictions.parquet")


if __name__ == '__main__':
    main()
