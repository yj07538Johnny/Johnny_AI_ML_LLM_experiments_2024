#!/usr/bin/env python3
"""
cnn_feature_extractor_ngram.py

Extract features from n-gram embeddings using CNN for MOE/MOP tag classification.
Designed to be run iteratively for each n-gram column (ngram_1, ngram_2, ..., ngram_100).

Usage:
    python cnn_feature_extractor_ngram.py \
        --input data.parquet \
        --output features.parquet \
        --ngram-col NGRAM_COL \
        --embedding-col NGRAM_EMBEDDING_COL \
        --tag-col MOE_MOP_TAG_COL

Example iteration:
    for n in 1 2 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100; do
        python cnn_feature_extractor_ngram.py \
            --input data.parquet \
            --output data.parquet \
            --ngram-col "ngram_${n}" \
            --embedding-col "ngram_${n}_embedding" \
            --tag-col moe_mop_tag
    done

Generic column names to replace:
    - NGRAM_COL: Column containing list of n-grams
    - NGRAM_EMBEDDING_COL: Column containing list of embeddings for the n-grams
    - MOE_MOP_TAG_COL: Column containing MOE/MOP tags (single tag or empty)
"""

import argparse
import os
import numpy as np
import pandas as pd
import duckdb
import torch
from pathlib import Path
import json

from cnn_feature_extractor_base import (
    EmbeddingDataset,
    CNNFeatureExtractor,
    prepare_data_for_training,
    train_model,
    extract_features_for_all_rows
)
from torch.utils.data import DataLoader


def load_data(input_path: str) -> pd.DataFrame:
    """Load data from parquet file (via DuckDB or directly)."""
    input_path = Path(input_path)

    if input_path.suffix == '.duckdb':
        con = duckdb.connect(str(input_path))
        tables = con.execute("SHOW TABLES").fetchall()
        if not tables:
            raise ValueError("No tables found in DuckDB database")
        table_name = tables[0][0]
        df = con.execute(f"SELECT * FROM {table_name}").df()
        con.close()
    elif input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        con = duckdb.connect()
        df = con.execute(f"SELECT * FROM '{input_path}'").df()
        con.close()

    return df


def save_data(df: pd.DataFrame, output_path: str):
    """Save DataFrame to parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"Saved output to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract features from n-gram embeddings using CNN'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input parquet/duckdb file path'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output parquet file path'
    )
    parser.add_argument(
        '--ngram-col',
        default='NGRAM_COL',
        help='Column name containing n-grams (list of strings)'
    )
    parser.add_argument(
        '--embedding-col',
        default='NGRAM_EMBEDDING_COL',
        help='Column name containing n-gram embeddings (list of vectors)'
    )
    parser.add_argument(
        '--tag-col',
        default='MOE_MOP_TAG_COL',
        help='Column name containing MOE/MOP tags'
    )
    parser.add_argument(
        '--output-feature-col',
        default=None,
        help='Name for output feature column (default: {ngram-col}_features)'
    )
    parser.add_argument(
        '--model-save-path',
        default=None,
        help='Path to save trained model (optional)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    parser.add_argument(
        '--num-filters',
        type=int,
        default=128,
        help='Number of CNN filters'
    )
    parser.add_argument(
        '--max-seq-len',
        type=int,
        default=None,
        help='Maximum sequence length (auto-detected if not specified)'
    )

    args = parser.parse_args()

    # Set default output feature column name
    if args.output_feature_col is None:
        args.output_feature_col = f"{args.ngram_col}_features"

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load data
    print(f"Loading data from: {args.input}")
    df = load_data(args.input)
    print(f"Loaded {len(df)} rows")

    # Check columns exist
    if args.embedding_col not in df.columns:
        raise ValueError(f"Embedding column '{args.embedding_col}' not found. "
                        f"Available columns: {df.columns.tolist()}")
    if args.tag_col not in df.columns:
        raise ValueError(f"Tag column '{args.tag_col}' not found. "
                        f"Available columns: {df.columns.tolist()}")
    if args.ngram_col not in df.columns:
        print(f"Warning: N-gram column '{args.ngram_col}' not found. "
              f"Proceeding with embedding column only.")

    # Prepare training data
    print(f"\nPreparing training data for {args.ngram_col}...")
    train_emb, train_labels, test_emb, test_labels, label_encoder, labeled_indices = \
        prepare_data_for_training(df, args.embedding_col, args.tag_col)

    # Determine dimensions
    embed_dim = None
    max_seq_len = 0
    for emb in train_emb:
        if emb is not None and len(emb) > 0:
            if isinstance(emb[0], (list, np.ndarray)):
                embed_dim = len(emb[0])
            else:
                embed_dim = len(emb)
            max_seq_len = max(max_seq_len, len(emb))

    if args.max_seq_len:
        max_seq_len = args.max_seq_len

    print(f"Embedding dimension: {embed_dim}")
    print(f"Max sequence length: {max_seq_len}")

    # Create datasets
    train_dataset = EmbeddingDataset(train_emb, train_labels, max_seq_len)
    test_dataset = EmbeddingDataset(test_emb, test_labels, max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    num_classes = len(label_encoder.classes_)
    model = CNNFeatureExtractor(
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_filters=args.num_filters,
        kernel_sizes=[3, 4, 5],
        dropout=0.5,
        fc_hidden=256
    )

    print(f"\nTraining CNN for {args.ngram_col}...")

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=device
    )

    print(f"Best test accuracy: {history['best_test_acc']:.4f}")

    # Save model if requested
    if args.model_save_path:
        model_path = Path(args.model_save_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'label_encoder_classes': label_encoder.classes_.tolist(),
            'embed_dim': embed_dim,
            'max_seq_len': max_seq_len,
            'num_classes': num_classes,
            'ngram_col': args.ngram_col,
            'history': history
        }, model_path)
        print(f"Model saved to: {model_path}")

    # Extract features for all rows
    print("\nExtracting features for all rows...")
    features = extract_features_for_all_rows(
        model=model,
        df=df,
        embedding_col=args.embedding_col,
        max_seq_len=max_seq_len,
        batch_size=args.batch_size,
        device=device
    )

    # Add feature column
    df[args.output_feature_col] = [feat.tolist() for feat in features]

    # Add predictions
    model.eval()
    all_preds = []

    with torch.no_grad():
        for i in range(0, len(df), args.batch_size):
            batch_features = features[i:i + args.batch_size]
            batch_tensor = torch.from_numpy(batch_features).to(device)
            logits = model.fc2(model.dropout(batch_tensor))
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())

    df[f'{args.output_feature_col}_pred'] = [
        label_encoder.inverse_transform([p])[0] for p in all_preds
    ]

    # Save
    save_data(df, args.output)

    print(f"\nN-gram feature extraction complete for {args.ngram_col}!")
    print(f"  Feature dimension: {features.shape[1]}")
    print(f"  New columns: {args.output_feature_col}, {args.output_feature_col}_pred")


if __name__ == '__main__':
    main()
