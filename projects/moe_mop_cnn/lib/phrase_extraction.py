"""
Phrase Extraction Module
========================

NLTK RegexpParser-based extraction of syntactic phrase types from text.
Each extractor uses POS-tagged tokens and a chunk grammar to identify
phrase constituents.

Supported phrase types:
    NP   - Noun phrases        (DT? JJ* NN+)
    VP   - Verb phrases        (VB* RB? VB* NNP? NN?)
    ADJP - Adjective phrases   (JJ+)
    ADVP - Adverb phrases      (RB+)
    PP   - Prepositional phrases (IN NP)
    GERUND  - Gerund phrases   (VBG NNP*)
    INF     - Infinitive phrases (TO VB*)
    PARTP   - Participle phrases (VBG|VBN NNP*)
    APPOS   - Appositive phrases (NNP NNP)
"""

from typing import List, Dict
from nltk import word_tokenize, pos_tag, RegexpParser


# Registry of phrase types and their grammars
PHRASE_GRAMMARS = {
    "noun_phrases":          ("NP",     r"NP: {<DT>?<JJ>*<NN.*>+}"),
    "verb_phrases":          ("VP",     r"VP: {<VB.*><RB>?<VB.*><NNP.?>*<NN.?>?}"),
    "adjective_phrases":     ("ADJP",   r"ADJP: {<JJ.*>+}"),
    "adverb_phrases":        ("ADVP",   r"ADVP: {<RB.*>+}"),
    "prepositional_phrases": ("PP",     r"PP: {<IN><NP>}"),
    "gerund_phrases":        ("GERUND", r"GERUND: {<VBG><NNP.*>?}"),
    "infinitive_phrases":    ("INF",    r"INF: {<TO><VB.*>}"),
    "participle_phrases":    ("PARTP",  r"PARTP: {<VBG|VBN><NNP.*>?}"),
    "appositive_phrases":    ("APPOS",  r"APPOS: {<NNP><NNP>}"),
}

PHRASE_TYPES = list(PHRASE_GRAMMARS.keys())


def _extract_phrases(text: str, label: str, grammar: str) -> List[str]:
    """Generic phrase extraction using NLTK RegexpParser.
    
    Args:
        text: Input text string
        label: Chunk label to extract (e.g., 'NP', 'VP')
        grammar: RegexpParser grammar string
        
    Returns:
        List of extracted phrase strings
    """
    if not text or not isinstance(text, str):
        return []
    
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    chunk_parser = RegexpParser(grammar)
    tree = chunk_parser.parse(pos_tags)
    
    phrases = []
    for subtree in tree.subtrees():
        if subtree.label() == label:
            phrase = " ".join(word for word, pos in subtree.leaves())
            phrases.append(phrase)
    return phrases


def extract_noun_phrases(text: str) -> List[str]:
    """Extract noun phrases from text."""
    label, grammar = PHRASE_GRAMMARS["noun_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_verb_phrases(text: str) -> List[str]:
    """Extract verb phrases from text."""
    label, grammar = PHRASE_GRAMMARS["verb_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_adjective_phrases(text: str) -> List[str]:
    """Extract adjective phrases from text."""
    label, grammar = PHRASE_GRAMMARS["adjective_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_adverb_phrases(text: str) -> List[str]:
    """Extract adverb phrases from text."""
    label, grammar = PHRASE_GRAMMARS["adverb_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_prepositional_phrases(text: str) -> List[str]:
    """Extract prepositional phrases from text."""
    label, grammar = PHRASE_GRAMMARS["prepositional_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_gerund_phrases(text: str) -> List[str]:
    """Extract gerund phrases from text."""
    label, grammar = PHRASE_GRAMMARS["gerund_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_infinitive_phrases(text: str) -> List[str]:
    """Extract infinitive phrases from text."""
    label, grammar = PHRASE_GRAMMARS["infinitive_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_participle_phrases(text: str) -> List[str]:
    """Extract participle phrases from text."""
    label, grammar = PHRASE_GRAMMARS["participle_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_appositive_phrases(text: str) -> List[str]:
    """Extract appositive phrases from text."""
    label, grammar = PHRASE_GRAMMARS["appositive_phrases"]
    return _extract_phrases(text, label, grammar)


def extract_all_phrases(text: str) -> Dict[str, List[str]]:
    """Extract all phrase types from text.
    
    Args:
        text: Input text string
        
    Returns:
        Dict mapping phrase type names to lists of extracted phrases
    """
    results = {}
    for phrase_type, (label, grammar) in PHRASE_GRAMMARS.items():
        results[phrase_type] = _extract_phrases(text, label, grammar)
    return results
