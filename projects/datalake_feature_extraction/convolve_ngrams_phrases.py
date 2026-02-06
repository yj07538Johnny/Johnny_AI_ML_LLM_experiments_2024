#!/usr/bin/env python3
"""
convolve_ngrams_phrases.py

Convolve predictive n-grams with phrases to find which combinations
are associated with MOE/MOP tags.

For each row:
1. Identify n-grams above the predictability threshold
2. Check which phrase types contain these predictive n-grams
3. Record the intersections as new features

Output columns:
- {ngram}_predictive_in_{phrase_type}: List of (ngram, score) tuples found in that phrase type
- {ngram}_phrase_intersections: Summary of all intersections for that n-gram size

Usage:
    python convolve_ngrams_phrases.py \
        --input data_with_scores.parquet \
        --output data_with_convolutions.parquet \
        --threshold 0.7 \
        --tag-col Narratives_sim_top_tag
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from collections import defaultdict
from tqdm import tqdm
import re
import warnings
warnings.filterwarnings('ignore')


def normalize_text(text):
    """Normalize text for comparison."""
    if text is None:
        return ""
    if isinstance(text, tuple):
        text = ' '.join(text)
    return str(text).lower().strip()


def ngram_in_phrase(ngram, phrase):
    """Check if an n-gram appears in a phrase (substring match)."""
    ngram_norm = normalize_text(ngram)
    phrase_norm = normalize_text(phrase)

    if not ngram_norm or not phrase_norm:
        return False

    return ngram_norm in phrase_norm or phrase_norm in ngram_norm


def ngram_overlaps_phrase(ngram, phrase, min_overlap=0.5):
    """
    Check if n-gram overlaps with phrase by word overlap.

    Args:
        ngram: The n-gram text
        phrase: The phrase text
        min_overlap: Minimum fraction of words that must overlap

    Returns:
        True if overlap meets threshold
    """
    ngram_words = set(normalize_text(ngram).split())
    phrase_words = set(normalize_text(phrase).split())

    if not ngram_words or not phrase_words:
        return False

    overlap = ngram_words & phrase_words

    # Check overlap ratio relative to smaller set
    smaller_size = min(len(ngram_words), len(phrase_words))
    overlap_ratio = len(overlap) / smaller_size if smaller_size > 0 else 0

    return overlap_ratio >= min_overlap


def find_phrase_intersections(ngrams, ngram_scores, phrases, match_mode='overlap'):
    """
    Find which n-grams intersect with which phrases.

    Args:
        ngrams: List of n-grams
        ngram_scores: List of predictability scores (aligned with ngrams)
        phrases: List of phrases to check against
        match_mode: 'exact' (substring), 'overlap' (word overlap), or 'both'

    Returns:
        List of (ngram, score, matched_phrase) tuples
    """
    if ngrams is None or phrases is None:
        return []

    if isinstance(ngrams, str):
        ngrams = [ngrams]
    if isinstance(phrases, str):
        phrases = [phrases]

    if ngram_scores is None or len(ngram_scores) == 0:
        return []

    intersections = []

    for i, (ngram, score) in enumerate(zip(ngrams, ngram_scores)):
        if i >= len(ngram_scores):
            break

        for phrase in phrases:
            matched = False

            if match_mode in ['exact', 'both']:
                if ngram_in_phrase(ngram, phrase):
                    matched = True

            if match_mode in ['overlap', 'both'] and not matched:
                if ngram_overlaps_phrase(ngram, phrase, min_overlap=0.5):
                    matched = True

            if matched:
                ngram_str = ' '.join(ngram) if isinstance(ngram, tuple) else str(ngram)
                phrase_str = ' '.join(phrase) if isinstance(phrase, tuple) else str(phrase)
                intersections.append({
                    'ngram': ngram_str,
                    'score': float(score),
                    'phrase': phrase_str
                })

    return intersections


def process_row(row, ngram_cols, phrase_cols, threshold, match_mode='overlap'):
    """
    Process a single row to find all n-gram/phrase intersections.

    Returns dict mapping:
        '{ngram_col}_in_{phrase_col}' -> list of intersections above threshold
    """
    results = {}

    for ngram_col in ngram_cols:
        score_col = f'{ngram_col}_predictability'

        ngrams = row.get(ngram_col)
        scores = row.get(score_col)

        if ngrams is None or scores is None:
            continue

        # Filter to predictive n-grams (above threshold)
        predictive_ngrams = []
        predictive_scores = []

        if isinstance(ngrams, str):
            ngrams = [ngrams]

        for i, (ng, sc) in enumerate(zip(ngrams, scores)):
            if sc >= threshold:
                predictive_ngrams.append(ng)
                predictive_scores.append(sc)

        if not predictive_ngrams:
            continue

        # Check against each phrase type
        for phrase_col in phrase_cols:
            phrases = row.get(phrase_col)

            if phrases is None:
                continue

            intersections = find_phrase_intersections(
                predictive_ngrams, predictive_scores, phrases, match_mode
            )

            key = f'{ngram_col}_in_{phrase_col}'
            results[key] = intersections

    return results


def aggregate_intersections(df, intersection_cols, tag_col):
    """
    Aggregate intersection patterns across all rows to find
    which n-gram/phrase combinations are most predictive.
    """
    # Count patterns by tag
    patterns_by_tag = defaultdict(lambda: defaultdict(list))

    for idx in range(len(df)):
        tag = df.iloc[idx].get(tag_col)
        if pd.isna(tag) or tag not in ['MOE', 'MOP']:
            continue

        for col in intersection_cols:
            intersections = df.iloc[idx].get(col)
            if intersections:
                for inter in intersections:
                    pattern = (inter['ngram'], inter['phrase'])
                    patterns_by_tag[tag][pattern].append(inter['score'])

    # Compute statistics per pattern
    pattern_stats = []

    all_patterns = set()
    for tag_patterns in patterns_by_tag.values():
        all_patterns.update(tag_patterns.keys())

    for pattern in all_patterns:
        ngram, phrase = pattern

        moe_scores = patterns_by_tag['MOE'].get(pattern, [])
        mop_scores = patterns_by_tag['MOP'].get(pattern, [])

        moe_count = len(moe_scores)
        mop_count = len(mop_scores)
        total_count = moe_count + mop_count

        if total_count == 0:
            continue

        # Compute discriminative score (how much does this pattern favor one class?)
        moe_ratio = moe_count / total_count
        mop_ratio = mop_count / total_count
        discriminative_score = abs(moe_ratio - mop_ratio)

        # Determine dominant class
        dominant_class = 'MOE' if moe_count > mop_count else 'MOP'

        pattern_stats.append({
            'ngram': ngram,
            'phrase': phrase,
            'moe_count': moe_count,
            'mop_count': mop_count,
            'total_count': total_count,
            'moe_ratio': moe_ratio,
            'mop_ratio': mop_ratio,
            'discriminative_score': discriminative_score,
            'dominant_class': dominant_class,
            'avg_predictability': np.mean(moe_scores + mop_scores) if (moe_scores + mop_scores) else 0
        })

    return pd.DataFrame(pattern_stats)


def main():
    parser = argparse.ArgumentParser(
        description='Convolve predictive n-grams with phrases'
    )
    parser.add_argument('--input', '-i', required=True, help='Input parquet with predictability scores')
    parser.add_argument('--output', '-o', required=True, help='Output parquet file')
    parser.add_argument('--threshold', '-t', type=float, default=0.7,
                       help='Predictability threshold (default: 0.7)')
    parser.add_argument('--tag-col', default='Narratives_sim_top_tag', help='Tag column')
    parser.add_argument('--match-mode', choices=['exact', 'overlap', 'both'], default='overlap',
                       help='How to match n-grams to phrases')
    parser.add_argument('--report-dir', default=None, help='Directory for analysis reports')

    args = parser.parse_args()

    # Load data
    print(f"Loading: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} rows")
    print(f"Using threshold: {args.threshold}")

    # Identify n-gram columns (those with _predictability scores)
    pred_cols = [c for c in df.columns if c.endswith('_predictability')]
    ngram_cols = [c.replace('_predictability', '') for c in pred_cols]

    print(f"\nN-gram columns with predictability scores ({len(ngram_cols)}):")
    for col in ngram_cols:
        print(f"  - {col}")

    # Identify phrase columns
    phrase_types = [
        'noun_phrases', 'verb_phrases', 'adjective_phrases', 'adverb_phrases',
        'prepositional_phrases', 'gerund_phrases', 'infinitive_phrases',
        'participle_phrases', 'appositive_phrases'
    ]
    phrase_cols = [p for p in phrase_types if p in df.columns]

    print(f"\nPhrase columns ({len(phrase_cols)}):")
    for col in phrase_cols:
        print(f"  - {col}")

    if not ngram_cols or not phrase_cols:
        print("ERROR: Need both n-gram and phrase columns")
        return

    # Process each row
    print(f"\nFinding n-gram/phrase intersections (threshold={args.threshold})...")

    all_intersection_cols = set()
    intersection_data = []

    for idx in tqdm(range(len(df)), desc="Processing rows"):
        row = df.iloc[idx]
        results = process_row(row, ngram_cols, phrase_cols, args.threshold, args.match_mode)
        intersection_data.append(results)
        all_intersection_cols.update(results.keys())

    # Add intersection columns to dataframe
    for col in all_intersection_cols:
        df[col] = [row.get(col, []) for row in intersection_data]

    print(f"\nNew intersection columns added ({len(all_intersection_cols)}):")
    for col in sorted(all_intersection_cols)[:10]:
        print(f"  - {col}")
    if len(all_intersection_cols) > 10:
        print(f"  ... and {len(all_intersection_cols) - 10} more")

    # Aggregate and analyze patterns
    print("\nAggregating intersection patterns...")
    pattern_df = aggregate_intersections(df, list(all_intersection_cols), args.tag_col)

    if len(pattern_df) > 0:
        # Sort by discriminative score
        pattern_df = pattern_df.sort_values('discriminative_score', ascending=False)

        print(f"\n{'='*80}")
        print("TOP DISCRIMINATIVE N-GRAM/PHRASE COMBINATIONS")
        print(f"{'='*80}")
        print(f"\n{'N-gram':<35} {'Phrase':<25} {'MOE':>5} {'MOP':>5} {'Discrim':>8} {'Dominant':<8}")
        print("-"*80)

        for _, row in pattern_df.head(30).iterrows():
            ngram_short = row['ngram'][:32] + '...' if len(row['ngram']) > 35 else row['ngram']
            phrase_short = row['phrase'][:22] + '...' if len(row['phrase']) > 25 else row['phrase']
            print(f"{ngram_short:<35} {phrase_short:<25} {row['moe_count']:>5} {row['mop_count']:>5} "
                  f"{row['discriminative_score']:>8.3f} {row['dominant_class']:<8}")

        # Save pattern report
        if args.report_dir:
            report_dir = Path(args.report_dir)
            report_dir.mkdir(parents=True, exist_ok=True)

            pattern_df.to_csv(report_dir / 'ngram_phrase_patterns.csv', index=False)

            # Save top patterns as JSON for easy inspection
            top_patterns = pattern_df.head(100).to_dict('records')
            with open(report_dir / 'top_patterns.json', 'w') as f:
                json.dump(top_patterns, f, indent=2)

            print(f"\nReports saved to: {report_dir}/")

    # Create summary columns
    # For each row, summarize: how many predictive n-grams found in each phrase type
    print("\nCreating summary columns...")

    summary_data = defaultdict(list)

    for idx in range(len(df)):
        phrase_type_counts = defaultdict(int)
        phrase_type_scores = defaultdict(list)

        for col in all_intersection_cols:
            intersections = df.iloc[idx].get(col, [])
            if intersections:
                # Extract phrase type from column name
                parts = col.split('_in_')
                if len(parts) == 2:
                    phrase_type = parts[1]
                    phrase_type_counts[phrase_type] += len(intersections)
                    phrase_type_scores[phrase_type].extend([i['score'] for i in intersections])

        # Store counts and average scores per phrase type
        for phrase_type in phrase_cols:
            count_col = f'predictive_ngrams_in_{phrase_type}_count'
            score_col = f'predictive_ngrams_in_{phrase_type}_avg_score'

            summary_data[count_col].append(phrase_type_counts.get(phrase_type, 0))

            scores = phrase_type_scores.get(phrase_type, [])
            avg_score = np.mean(scores) if scores else 0.0
            summary_data[score_col].append(avg_score)

    for col, values in summary_data.items():
        df[col] = values

    # Save output
    print(f"\nSaving to: {args.output}")
    df.to_parquet(args.output, index=False)

    # Final summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Threshold used: {args.threshold}")
    print(f"Match mode: {args.match_mode}")
    print(f"Intersection columns added: {len(all_intersection_cols)}")
    print(f"Summary columns added: {len(summary_data)}")
    print(f"Unique patterns found: {len(pattern_df)}")

    if len(pattern_df) > 0:
        highly_discriminative = pattern_df[pattern_df['discriminative_score'] >= 0.5]
        print(f"Highly discriminative patterns (score >= 0.5): {len(highly_discriminative)}")

        moe_dominant = pattern_df[pattern_df['dominant_class'] == 'MOE']
        mop_dominant = pattern_df[pattern_df['dominant_class'] == 'MOP']
        print(f"MOE-dominant patterns: {len(moe_dominant)}")
        print(f"MOP-dominant patterns: {len(mop_dominant)}")


if __name__ == '__main__':
    main()
