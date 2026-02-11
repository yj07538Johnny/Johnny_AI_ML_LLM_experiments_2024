#!/usr/bin/env python
# coding: utf-8
"""
WAR Report Feature Extraction Pipeline
=======================================

Main processing script for extracting features from Weekly Activity Reports,
training CNN classifiers, and extracting feature importance.

This replaces the monolithic Jupyter notebook with a clean, modular workflow
that uses the war_report_toolkit library.

Workflow:
    1. Initialize data store (DuckDB/Parquet)
    2. Check for new WAR file directories
    3. Process new files and append to store
    4. Load narrative data into working DataFrame
    5. Extract phrases and n-grams
    6. Vectorize features using Word2Vec (with exclusion list)
    7. Train CNN classifiers per feature type
    8. Extract feature importance scores
    9. Save results back to Parquet

Usage:
    python process_war_reports.py --data-dir ./data --war-dir /path/to/war_files
    python process_war_reports.py --data-dir ./data --extract-only    # Just extract features, no training
    python process_war_reports.py --data-dir ./data --train-only      # Train on existing features
"""

import os
import sys
import argparse
from pathlib import Path

# Add library to path
sys.path.insert(0, str(Path(__file__).parent))

from lib import (
    # Data store
    WarReportStore,
    # Phrase extraction
    extract_all_phrases, PHRASE_TYPES,
    # N-gram extraction
    get_bigrams, get_ngrams, NGRAM_SIZES,
    # Vectorization
    load_word2vec_model, get_phrase_vectors, get_ngram_vectors,
    # CNN
    prepare_data, create_model, train_model, get_feature_importance,
    process_phrases_ngrams,
    # Exclusion list
    ExclusionList,
    # Utils
    apply_with_progress,
)
from lib.symbolic_extraction import extract_all_symbolic, load_models


def parse_args():
    parser = argparse.ArgumentParser(description="WAR Report Feature Extraction Pipeline")
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Directory for Parquet data files")
    parser.add_argument("--war-dir", type=str, default=None,
                        help="Root directory containing WAR file subdirectories")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to pretrained Word2Vec model")
    parser.add_argument("--exclusions-file", type=str, default=None,
                        help="Path to exclusion list JSON file")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract features, skip CNN training")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train CNN on existing features")
    parser.add_argument("--symbolic", action="store_true",
                        help="Also run symbolic extraction (NER, POS, keywords)")
    parser.add_argument("--ngram-sizes", type=int, nargs="+", default=None,
                        help="N-gram sizes to extract (default: standard set)")
    return parser.parse_args()


def step_1_initialize(args) -> WarReportStore:
    """Initialize data store and show status."""
    print("\n" + "="*60)
    print("STEP 1: Initialize Data Store")
    print("="*60)
    
    store = WarReportStore(args.data_dir)
    print(store)
    
    status = store.status()
    print(f"  Data directory: {status['data_dir']}")
    print(f"  Reports: {status.get('report_count', 0)}")
    print(f"  Features: {status.get('feature_count', 0)}")
    print(f"  Processed directories: {status['processed_directories']}")
    
    return store


def step_2_check_new_files(store: WarReportStore, war_dir: str):
    """Check for and process new WAR file directories."""
    print("\n" + "="*60)
    print("STEP 2: Check for New Files")
    print("="*60)
    
    if not war_dir:
        print("  No WAR directory specified, skipping file check")
        return
    
    new_dirs = store.get_unprocessed_directories(war_dir)
    
    if not new_dirs:
        print("  No new directories to process")
        return
    
    print(f"  Found {len(new_dirs)} new directories:")
    for d in new_dirs:
        print(f"    {d}")
    
    # Process each new directory
    # This is where create_path_df() logic would go
    # For now, log them as processed
    for d in new_dirs:
        # TODO: Call your create_path_df() equivalent here
        # new_df = create_path_df_from_directory(d)
        # store.append_reports(new_df)
        
        file_count = len(list(Path(d).glob("*")))
        store.log_processed_directory(d, file_count)
        print(f"  Processed: {d} ({file_count} files)")


def step_3_load_data(store: WarReportStore):
    """Load data from Parquet into working DataFrames."""
    print("\n" + "="*60)
    print("STEP 3: Load Data")
    print("="*60)
    
    report_df = store.load_reports()
    feature_df = store.load_features()
    
    print(f"  Reports loaded: {len(report_df)} rows")
    print(f"  Features loaded: {len(feature_df)} rows")
    
    # Filter to MOE/MOP if features exist
    if len(feature_df) > 0 and 'Narratives_sim_top_tag' in feature_df.columns:
        filtered_df = feature_df[
            feature_df['Narratives_sim_top_tag'].isin(['MOE', 'MOP'])
        ].reset_index(drop=True)
        print(f"  MOE/MOP filtered: {len(filtered_df)} rows")
    else:
        filtered_df = feature_df
    
    return report_df, feature_df, filtered_df


def step_4_extract_phrases(filtered_df):
    """Extract all phrase types from narrative text."""
    print("\n" + "="*60)
    print("STEP 4: Extract Phrases")
    print("="*60)
    
    if 'narrative' not in filtered_df.columns:
        print("  No 'narrative' column found, skipping phrase extraction")
        return filtered_df
    
    for phrase_type in PHRASE_TYPES:
        col_name = phrase_type
        if col_name not in filtered_df.columns:
            from lib.phrase_extraction import PHRASE_GRAMMARS, _extract_phrases
            label, grammar = PHRASE_GRAMMARS[phrase_type]
            print(f"  Extracting {phrase_type}...")
            filtered_df[col_name] = apply_with_progress(
                filtered_df['narrative'],
                lambda text, l=label, g=grammar: _extract_phrases(text, l, g),
                desc=f"Extracting {phrase_type}"
            )
        else:
            print(f"  {phrase_type} already extracted, skipping")
    
    return filtered_df


def step_5_extract_ngrams(filtered_df, ngram_sizes=None):
    """Extract n-grams from narrative text."""
    print("\n" + "="*60)
    print("STEP 5: Extract N-grams")
    print("="*60)
    
    if 'narrative' not in filtered_df.columns:
        print("  No 'narrative' column found, skipping n-gram extraction")
        return filtered_df
    
    sizes = ngram_sizes or NGRAM_SIZES
    
    for n in sizes:
        col_name = "bigrams" if n == 2 else f"{n}-grams"
        if col_name not in filtered_df.columns:
            print(f"  Extracting {col_name}...")
            filtered_df[col_name] = apply_with_progress(
                filtered_df['narrative'],
                lambda text, size=n: get_ngrams(text, size),
                desc=f"Extracting {col_name}"
            )
        else:
            print(f"  {col_name} already extracted, skipping")
    
    return filtered_df


def step_6_vectorize(filtered_df, model_path: str, exclusions: ExclusionList = None):
    """Vectorize all extracted features using Word2Vec."""
    print("\n" + "="*60)
    print("STEP 6: Vectorize Features")
    print("="*60)
    
    if not model_path:
        print("  No Word2Vec model path specified, skipping vectorization")
        return filtered_df
    
    print(f"  Loading Word2Vec model from {model_path}")
    model = load_word2vec_model(model_path)
    print(f"  Model loaded: {model.wv.vector_size}-dimensional, {len(model.wv)} vocabulary")
    
    if exclusions:
        print(f"  Exclusion list: {len(exclusions)} terms, mode={exclusions.mode}")
    
    # Vectorize phrase columns
    for phrase_type in PHRASE_TYPES:
        vec_col = f"vector_{phrase_type}"
        if phrase_type in filtered_df.columns and vec_col not in filtered_df.columns:
            print(f"  Vectorizing {phrase_type}...")
            filtered_df[vec_col] = apply_with_progress(
                filtered_df[phrase_type],
                get_phrase_vectors,
                desc=f"Vectorizing {phrase_type}",
                model=model,
                exclusions=exclusions,
            )
    
    # Vectorize n-gram columns
    ngram_cols = [c for c in filtered_df.columns 
                  if c.endswith('-grams') or c == 'bigrams']
    
    for col in ngram_cols:
        vec_col = f"vector_{col.replace('-', '_')}"
        if vec_col not in filtered_df.columns:
            print(f"  Vectorizing {col}...")
            filtered_df[vec_col] = apply_with_progress(
                filtered_df[col],
                get_ngram_vectors,
                desc=f"Vectorizing {col}",
                model=model,
                exclusions=exclusions,
            )
    
    return filtered_df


def step_7_train_and_extract(filtered_df):
    """Train CNN classifiers and extract feature importance."""
    print("\n" + "="*60)
    print("STEP 7: Train CNN & Extract Feature Importance")
    print("="*60)
    
    if 'Narratives_sim_top_tag' not in filtered_df.columns:
        print("  No label column found, skipping training")
        return filtered_df
    
    labels_col = 'Narratives_sim_top_tag'
    
    # Process phrase columns
    phrase_columns = [pt for pt in PHRASE_TYPES if pt in filtered_df.columns]
    for col in phrase_columns:
        vec_col = f"vector_{col}"
        tokens_col = col
        if vec_col in filtered_df.columns:
            print(f"  Training CNN on {col}...")
            filtered_df = process_phrases_ngrams(
                filtered_df, tokens_col, vec_col, labels_col,
                phrase_type=col
            )
    
    # Process n-gram columns
    ngram_cols = [c for c in filtered_df.columns 
                  if (c.endswith('-grams') or c == 'bigrams') and not c.startswith('vector_')]
    
    for col in ngram_cols:
        vec_col = f"vector_{col.replace('-', '_')}"
        if vec_col in filtered_df.columns:
            print(f"  Training CNN on {col}...")
            filtered_df = process_phrases_ngrams(
                filtered_df, col, vec_col, labels_col,
                phrase_type=col
            )
    
    return filtered_df


def step_8_symbolic_extraction(filtered_df, models=None):
    """Run symbolic feature extraction (NER, POS, keywords)."""
    print("\n" + "="*60)
    print("STEP 8: Symbolic Feature Extraction")
    print("="*60)
    
    if models is None:
        print("  Loading NLP models...")
        models = load_models(['ner', 'pos', 'keywords'])
    
    if 'narrative' not in filtered_df.columns:
        print("  No 'narrative' column, skipping symbolic extraction")
        return filtered_df
    
    # NER
    if 'ner' not in filtered_df.columns:
        from lib.symbolic_extraction import extract_ner
        print("  Extracting named entities...")
        filtered_df['ner'] = apply_with_progress(
            filtered_df['narrative'],
            lambda text: extract_ner(text, models),
            desc="NER extraction"
        )
    
    # POS
    if 'pos' not in filtered_df.columns:
        from lib.symbolic_extraction import extract_pos
        print("  Extracting POS tags...")
        filtered_df['pos'] = apply_with_progress(
            filtered_df['narrative'],
            lambda text: extract_pos(text, models),
            desc="POS extraction"
        )
    
    # Keywords
    if 'keywords' not in filtered_df.columns:
        from lib.symbolic_extraction import extract_keywords
        print("  Extracting keywords...")
        filtered_df['keywords'] = apply_with_progress(
            filtered_df['narrative'],
            lambda text: extract_keywords(text, models),
            desc="Keyword extraction"
        )
    
    return filtered_df


def step_9_save(store: WarReportStore, filtered_df):
    """Save results back to Parquet."""
    print("\n" + "="*60)
    print("STEP 9: Save Results")
    print("="*60)
    
    store.save_features(filtered_df)
    print(f"  Saved {len(filtered_df)} records with {len(filtered_df.columns)} columns")
    print(f"  Columns: {list(filtered_df.columns)}")


def main():
    args = parse_args()
    
    # Load exclusion list
    exclusions = None
    if args.exclusions_file and os.path.exists(args.exclusions_file):
        exclusions = ExclusionList.load(args.exclusions_file)
        print(f"Loaded exclusion list: {exclusions}")
    else:
        exclusions = ExclusionList(use_defaults=True)
        print(f"Using default exclusion list: {exclusions}")
    
    # Pipeline
    store = step_1_initialize(args)
    
    if not args.train_only:
        step_2_check_new_files(store, args.war_dir)
    
    report_df, feature_df, filtered_df = step_3_load_data(store)
    
    if not args.train_only:
        filtered_df = step_4_extract_phrases(filtered_df)
        filtered_df = step_5_extract_ngrams(filtered_df, args.ngram_sizes)
        filtered_df = step_6_vectorize(filtered_df, args.model_path, exclusions)
    
    if not args.extract_only:
        filtered_df = step_7_train_and_extract(filtered_df)
    
    if args.symbolic:
        filtered_df = step_8_symbolic_extraction(filtered_df)
    
    step_9_save(store, filtered_df)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(store)


if __name__ == "__main__":
    main()
