"""
WAR Report Toolkit Library
==========================

Modular library for processing Weekly Activity Reports (WAR),
extracting features, training classifiers, and managing data persistence.

Modules:
    phrase_extraction  - NLTK-based syntactic phrase extractors
    ngram_extraction   - Sentence-boundary-respecting n-gram extraction
    vectorization      - Word2Vec-based vectorization for phrases and n-grams
    cnn_classifier     - 1D CNN for text classification and feature importance
    symbolic_extraction - AMR, FOL, POS, NER, keywords, ontology extraction
    data_store         - DuckDB/Parquet persistence layer
    exclusion_list     - Managed exclusion terms for vectorization
    utils              - Progress bars, data prep helpers
"""

from .phrase_extraction import (
    extract_noun_phrases,
    extract_verb_phrases,
    extract_adjective_phrases,
    extract_adverb_phrases,
    extract_prepositional_phrases,
    extract_gerund_phrases,
    extract_infinitive_phrases,
    extract_participle_phrases,
    extract_appositive_phrases,
    extract_all_phrases,
    PHRASE_TYPES,
)

from .ngram_extraction import (
    get_bigrams,
    get_ngrams,
    get_all_ngrams,
    NGRAM_SIZES,
)

from .vectorization import (
    get_word_vectors,
    combine_vectors_sum,
    get_phrase_vectors,
    get_ngram_vectors,
    vectorize_column,
    load_word2vec_model,
)

from .cnn_classifier import (
    prepare_data,
    create_model,
    train_model,
    get_feature_importance,
    process_phrases_ngrams,
)

from .exclusion_list import (
    ExclusionList,
    DEFAULT_EXCLUSIONS,
)

from .data_store import (
    WarReportStore,
)

from .utils import (
    apply_with_progress,
)

__version__ = '0.1.0'

__all__ = [
    # phrase_extraction
    'extract_noun_phrases', 'extract_verb_phrases', 'extract_adjective_phrases',
    'extract_adverb_phrases', 'extract_prepositional_phrases', 'extract_gerund_phrases',
    'extract_infinitive_phrases', 'extract_participle_phrases', 'extract_appositive_phrases',
    'extract_all_phrases', 'PHRASE_TYPES',
    # ngram_extraction
    'get_bigrams', 'get_ngrams', 'get_all_ngrams', 'NGRAM_SIZES',
    # vectorization
    'get_word_vectors', 'combine_vectors_sum', 'get_phrase_vectors',
    'get_ngram_vectors', 'vectorize_column', 'load_word2vec_model',
    # cnn_classifier
    'prepare_data', 'create_model', 'train_model',
    'get_feature_importance', 'process_phrases_ngrams',
    # exclusion_list
    'ExclusionList', 'DEFAULT_EXCLUSIONS',
    # data_store
    'WarReportStore',
    # utils
    'apply_with_progress',
]
