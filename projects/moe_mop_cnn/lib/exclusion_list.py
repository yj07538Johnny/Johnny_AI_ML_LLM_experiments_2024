"""
Exclusion List Management
=========================

Manages terms that should be excluded from vectorization and feature
importance scoring. These are terms that are highly predictive but not
meaningful — e.g., "National Security Agency" appears in nearly every
document and dominates feature importance without providing signal.

Two modes of exclusion:
    1. null_vector: Replace excluded terms with zero vectors when building
       document vectors (term still present in text, just neutralized)
    2. omit: Skip excluded terms entirely when building document vectors

Usage:
    exclusions = ExclusionList()
    exclusions.add("national security agency")
    exclusions.add("nsa")
    
    # Check if a token/phrase should be excluded
    if not exclusions.is_excluded("national security agency"):
        vectors.append(model.wv[token])
    
    # Use with vectorization
    vectors = get_word_vectors(tokens, model, exclusions=exclusions)
    
    # Persist to file
    exclusions.save("exclusions.json")
    exclusions = ExclusionList.load("exclusions.json")
"""

import json
import re
from typing import List, Set, Optional
from pathlib import Path


# Default exclusion terms — high-frequency domain terms that dominate
# feature importance without providing meaningful signal
DEFAULT_EXCLUSIONS = [
    "national security agency",
    "nsa",
    "united states",
    "u.s.",
    "us government",
    "classified",
    "unclassified",
    "fouo",
    "top secret",
    "secret",
    "confidential",
]


class ExclusionList:
    """Managed exclusion list for term filtering during vectorization.
    
    Supports exact match and pattern-based exclusion.
    All comparisons are case-insensitive.
    
    Attributes:
        terms: Set of excluded term strings (lowercased)
        patterns: List of compiled regex patterns for exclusion
        mode: 'null_vector' or 'omit' — how to handle excluded terms
    """
    
    def __init__(self, terms: Optional[List[str]] = None, 
                 mode: str = "null_vector",
                 use_defaults: bool = True):
        """Initialize exclusion list.
        
        Args:
            terms: Additional terms to exclude
            mode: 'null_vector' (replace with zeros) or 'omit' (skip entirely)
            use_defaults: Whether to include DEFAULT_EXCLUSIONS
        """
        self.mode = mode
        self.terms: Set[str] = set()
        self.patterns: list = []
        
        if use_defaults:
            for term in DEFAULT_EXCLUSIONS:
                self.terms.add(term.lower().strip())
        
        if terms:
            for term in terms:
                self.add(term)
    
    def add(self, term: str):
        """Add a term to the exclusion list."""
        self.terms.add(term.lower().strip())
    
    def add_pattern(self, pattern: str):
        """Add a regex pattern for exclusion."""
        self.patterns.append(re.compile(pattern, re.IGNORECASE))
    
    def remove(self, term: str):
        """Remove a term from the exclusion list."""
        self.terms.discard(term.lower().strip())
    
    def is_excluded(self, term: str) -> bool:
        """Check if a term should be excluded.
        
        Args:
            term: Token, phrase, or n-gram string to check
            
        Returns:
            True if the term matches any exclusion
        """
        normalized = term.lower().strip()
        
        # Exact match
        if normalized in self.terms:
            return True
        
        # Check if term contains any excluded term as substring
        for excluded in self.terms:
            if excluded in normalized:
                return True
        
        # Pattern match
        for pattern in self.patterns:
            if pattern.search(normalized):
                return True
        
        return False
    
    def filter_tokens(self, tokens: list) -> list:
        """Filter a list of tokens, removing excluded ones.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Filtered list with excluded tokens removed
        """
        return [t for t in tokens if not self.is_excluded(t)]
    
    def list_terms(self) -> List[str]:
        """Return sorted list of all excluded terms."""
        return sorted(self.terms)
    
    def save(self, path: str):
        """Save exclusion list to JSON file."""
        data = {
            "mode": self.mode,
            "terms": sorted(self.terms),
            "patterns": [p.pattern for p in self.patterns],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ExclusionList':
        """Load exclusion list from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        obj = cls(terms=data.get("terms", []), 
                  mode=data.get("mode", "null_vector"),
                  use_defaults=False)
        
        for pattern in data.get("patterns", []):
            obj.add_pattern(pattern)
        
        return obj
    
    def __len__(self):
        return len(self.terms)
    
    def __repr__(self):
        return f"ExclusionList(n_terms={len(self.terms)}, n_patterns={len(self.patterns)}, mode='{self.mode}')"
