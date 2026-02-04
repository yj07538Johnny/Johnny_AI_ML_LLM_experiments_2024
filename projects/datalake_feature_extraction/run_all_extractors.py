#!/usr/bin/env python3
"""
run_all_extractors.py

Batch runner to execute feature extraction on multiple columns.
Configure your column mappings and run this script to process all at once.

Usage:
    python run_all_extractors.py --config config.yaml
    python run_all_extractors.py --input data.parquet --tag-col moe_mop_tag

This script will:
1. Auto-detect columns with 'embedding' in the name
2. Run the appropriate extractor for each
3. Accumulate features in the output parquet file
"""

import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml
import re


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def detect_embedding_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect embedding columns and their types.

    Returns dict mapping embedding_col -> {'type': 'narrative'|'ngram'|'phrase', 'source_col': ...}
    """
    columns = {}
    embed_cols = [c for c in df.columns if 'embedding' in c.lower()]

    for col in embed_cols:
        col_lower = col.lower()

        # Try to find corresponding source column
        source_col = col.replace('_embedding', '').replace('_embed', '').replace('embedding_', '')

        if 'narrative' in col_lower:
            columns[col] = {'type': 'narrative', 'source_col': source_col}
        elif 'ngram' in col_lower or re.search(r'gram_?\d+', col_lower):
            columns[col] = {'type': 'ngram', 'source_col': source_col}
        elif 'phrase' in col_lower or any(x in col_lower for x in ['noun', 'verb', 'adj', 'prep']):
            columns[col] = {'type': 'phrase', 'source_col': source_col}
        else:
            # Default to phrase type
            columns[col] = {'type': 'phrase', 'source_col': source_col}

    return columns


def run_extractor(
    extractor_type: str,
    input_path: str,
    output_path: str,
    embedding_col: str,
    source_col: str,
    tag_col: str,
    **kwargs
):
    """Run a single extractor script."""
    script_dir = Path(__file__).parent

    if extractor_type == 'narrative':
        script = script_dir / 'cnn_feature_extractor_narrative.py'
        cmd = [
            sys.executable, str(script),
            '--input', input_path,
            '--output', output_path,
            '--embedding-col', embedding_col,
            '--tag-col', tag_col,
            '--output-feature-col', f'{source_col}_features'
        ]
    elif extractor_type == 'ngram':
        script = script_dir / 'cnn_feature_extractor_ngram.py'
        cmd = [
            sys.executable, str(script),
            '--input', input_path,
            '--output', output_path,
            '--ngram-col', source_col,
            '--embedding-col', embedding_col,
            '--tag-col', tag_col
        ]
    elif extractor_type == 'phrase':
        script = script_dir / 'cnn_feature_extractor_phrase.py'
        cmd = [
            sys.executable, str(script),
            '--input', input_path,
            '--output', output_path,
            '--phrase-col', source_col,
            '--embedding-col', embedding_col,
            '--tag-col', tag_col
        ]
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")

    # Add optional arguments
    if 'num_epochs' in kwargs:
        cmd.extend(['--num-epochs', str(kwargs['num_epochs'])])
    if 'batch_size' in kwargs:
        cmd.extend(['--batch-size', str(kwargs['batch_size'])])
    if 'model_save_dir' in kwargs and kwargs['model_save_dir']:
        model_path = Path(kwargs['model_save_dir']) / f'{source_col}_model.pt'
        cmd.extend(['--model-save-path', str(model_path)])

    print(f"\n{'='*60}")
    print(f"Running {extractor_type} extractor for: {embedding_col}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"WARNING: Extractor failed for {embedding_col}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Batch run feature extractors on all embedding columns'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input parquet file'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output parquet file (default: overwrites input)'
    )
    parser.add_argument(
        '--tag-col',
        required=True,
        help='Column containing MOE/MOP tags'
    )
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='YAML config file with column mappings (optional)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=50,
        help='Training epochs for each model'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--save-models',
        default=None,
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--only-type',
        choices=['narrative', 'ngram', 'phrase'],
        default=None,
        help='Only run extractors of this type'
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.input

    # Load data to detect columns
    print(f"Loading data from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Get column mappings
    if args.config:
        config = load_config(args.config)
        columns = config.get('columns', {})
    else:
        # Auto-detect
        columns = detect_embedding_columns(df)
        print(f"\nAuto-detected {len(columns)} embedding columns:")
        for col, info in columns.items():
            print(f"  {col} -> type={info['type']}, source={info['source_col']}")

    if not columns:
        print("No embedding columns found!")
        return

    # Process each column
    successful = 0
    failed = 0
    current_input = args.input

    for embedding_col, info in columns.items():
        col_type = info['type']
        source_col = info['source_col']

        if args.only_type and col_type != args.only_type:
            print(f"Skipping {embedding_col} (type={col_type}, only processing {args.only_type})")
            continue

        success = run_extractor(
            extractor_type=col_type,
            input_path=current_input,
            output_path=args.output,
            embedding_col=embedding_col,
            source_col=source_col,
            tag_col=args.tag_col,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            model_save_dir=args.save_models
        )

        if success:
            successful += 1
            # Use output as input for next iteration (accumulate features)
            current_input = args.output
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output: {args.output}")
    print('='*60)


if __name__ == '__main__':
    main()
