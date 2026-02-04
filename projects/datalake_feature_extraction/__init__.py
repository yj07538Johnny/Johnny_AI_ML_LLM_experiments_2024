"""
Datalake Feature Extraction

CNN-based feature extraction for MOE/MOP tag classification from embeddings.

Modules:
    - pandas_to_duckdb_parquet: Convert pandas DataFrames to DuckDB/Parquet
    - cnn_feature_extractor_base: Shared CNN architecture and utilities
    - cnn_feature_extractor_narrative: Extract features from narrative embeddings
    - cnn_feature_extractor_ngram: Extract features from n-gram embeddings
    - cnn_feature_extractor_phrase: Extract features from phrase embeddings
    - run_all_extractors: Batch runner for all extractors
"""

from .pandas_to_duckdb_parquet import create_duckdb_from_dataframe, load_dataframe
from .cnn_feature_extractor_base import (
    EmbeddingDataset,
    CNNFeatureExtractor,
    prepare_data_for_training,
    train_model,
    extract_features_for_all_rows
)

__version__ = '0.1.0'
__all__ = [
    'create_duckdb_from_dataframe',
    'load_dataframe',
    'EmbeddingDataset',
    'CNNFeatureExtractor',
    'prepare_data_for_training',
    'train_model',
    'extract_features_for_all_rows'
]
