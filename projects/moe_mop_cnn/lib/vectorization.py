"""
Vectorization Module
====================

Word2Vec-based vectorization for tokens, phrases, and n-grams.
Supports exclusion lists for filtering domain-specific stopwords
that would otherwise dominate feature importance.

Exclusion modes:
    null_vector: Replace excluded terms with zero vectors
    omit: Skip excluded terms entirely
"""

import re
import numpy as np
from typing import List, Optional, Tuple, Union
from gensim.models import Word2Vec

from .exclusion_list import ExclusionList


def load_word2vec_model(model_path: str) -> Word2Vec:
    """Load a pretrained Word2Vec model.
    
    Args:
        model_path: Path to the Word2Vec model file
        
    Returns:
        Loaded Word2Vec model
    """
    return Word2Vec.load(model_path)


def get_word_vectors(tokens: list, model: Word2Vec, 
                     exclusions: Optional[ExclusionList] = None) -> Optional[np.ndarray]:
    """Get word vectors for a list of tokens.
    
    Args:
        tokens: List of token strings
        model: Pretrained Word2Vec model
        exclusions: Optional exclusion list for filtering terms
        
    Returns:
        numpy array of vectors, or None if no tokens found in vocabulary
    """
    vectors = []
    for token in tokens:
        # Check exclusion list
        if exclusions and exclusions.is_excluded(token):
            if exclusions.mode == "null_vector":
                vectors.append(np.zeros(model.vector_size))
            # mode == "omit": skip entirely
            continue
        
        if token in model.wv:
            vectors.append(model.wv[token])
    
    return np.array(vectors) if vectors else None


def combine_vectors_sum(tokens: list, model: Word2Vec,
                        exclusions: Optional[ExclusionList] = None) -> np.ndarray:
    """Combine token vectors by summing them.
    
    Args:
        tokens: List of token strings
        model: Pretrained Word2Vec model
        exclusions: Optional exclusion list
        
    Returns:
        Summed vector, or zero vector if no tokens in vocabulary
    """
    vectors = []
    for token in tokens:
        if exclusions and exclusions.is_excluded(token):
            if exclusions.mode == "null_vector":
                vectors.append(np.zeros(model.vector_size))
            continue
        
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if vectors:
        return np.sum(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


def get_phrase_vectors(phrases, model: Word2Vec,
                       exclusions: Optional[ExclusionList] = None) -> list:
    """Get vectors for a list of phrases by summing constituent token vectors.
    
    Args:
        phrases: List of phrase strings, or a single string
        model: Pretrained Word2Vec model
        exclusions: Optional exclusion list
        
    Returns:
        List of numpy vectors (one per phrase)
    """
    if isinstance(phrases, str):
        phrases = [phrases]
    
    vectors = []
    for phrase in phrases:
        from nltk import word_tokenize
        tokens = word_tokenize(phrase)
        tokens = [token.lower() for token in tokens]
        phrase_vector = combine_vectors_sum(tokens, model, exclusions)
        vectors.append(phrase_vector)
    
    return vectors


def get_ngram_vectors(ngrams_list, model: Word2Vec,
                      exclusions: Optional[ExclusionList] = None) -> list:
    """Get vectors for a list of n-grams.
    
    Handles both tuple n-grams and string n-grams.
    
    Args:
        ngrams_list: List of n-gram tuples or strings
        model: Pretrained Word2Vec model
        exclusions: Optional exclusion list
        
    Returns:
        List of numpy vectors (one per n-gram)
    """
    vectors = []
    
    if isinstance(ngrams_list, str):
        ngrams_list = [ngrams_list]
    
    for ngram in ngrams_list:
        if isinstance(ngram, tuple):
            tokens = [token.lower() for token in ngram]
        else:
            # String n-gram — parse tokens
            tokens = re.findall(r'\w+', str(ngram))
            tokens = [token.lower() for token in tokens]
        
        ngram_vector = combine_vectors_sum(tokens, model, exclusions)
        vectors.append(ngram_vector)
    
    return vectors


def vectorize_column(series, vectorize_func, model: Word2Vec,
                     exclusions: Optional[ExclusionList] = None,
                     desc: str = "") -> list:
    """Vectorize an entire DataFrame column using the specified function.
    
    Args:
        series: pandas Series containing phrases or n-grams
        vectorize_func: Function to use (get_phrase_vectors or get_ngram_vectors)
        model: Pretrained Word2Vec model
        exclusions: Optional exclusion list
        desc: Progress bar description
        
    Returns:
        List of vector lists (one per row)
    """
    from .utils import apply_with_progress
    return apply_with_progress(
        series, vectorize_func, desc=desc,
        model=model, exclusions=exclusions
    )
