"""
N-gram Extraction Module
=========================

Extracts n-grams from text while respecting sentence boundaries.
N-grams never cross sentence boundaries.

Supports configurable n-gram sizes from bigrams through arbitrary length.
"""

from typing import List, Tuple, Union
import nltk
from nltk import word_tokenize, ngrams as nltk_ngrams


# Default n-gram sizes to extract
NGRAM_SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]


def get_bigrams(paragraph: str) -> List[Tuple[str, ...]]:
    """Get bigrams from text within sentence boundaries.
    
    Args:
        paragraph: Input text
        
    Returns:
        List of bigram tuples
    """
    return get_ngrams(paragraph, n=2)


def get_ngrams(paragraph: str, n: int) -> List[Tuple[str, ...]]:
    """Get n-grams from text within sentence boundaries.
    
    N-grams never cross sentence boundaries.
    
    Args:
        paragraph: Input text
        n: N-gram size
        
    Returns:
        List of n-gram tuples
    """
    if not isinstance(paragraph, str) or not paragraph.strip():
        return []
    
    sentences = nltk.sent_tokenize(paragraph)
    result = []
    
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        if len(tokens) >= n:
            sentence_ngrams = list(nltk_ngrams(tokens, n))
            result.extend(sentence_ngrams)
    
    return result


def get_all_ngrams(paragraph: str, sizes: List[int] = None) -> dict:
    """Extract n-grams of all configured sizes.
    
    Args:
        paragraph: Input text
        sizes: List of n-gram sizes (defaults to NGRAM_SIZES)
        
    Returns:
        Dict mapping f"{n}-grams" to list of n-gram tuples
    """
    if sizes is None:
        sizes = NGRAM_SIZES
    
    results = {}
    for n in sizes:
        key = f"{n}-grams" if n != 2 else "bigrams"
        results[key] = get_ngrams(paragraph, n)
    return results
