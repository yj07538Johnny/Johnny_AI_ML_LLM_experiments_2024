"""
Symbolic Extraction Module
==========================

Feature extraction functions for symbolic/neurosymbolic processing.
These mirror the extraction library from the hallucination datalake project
and provide a consistent interface for extracting structured features
from text.

Each extractor returns a standardized format that can be stored in
the DuckDB/Parquet data store and used for downstream classification.

Note: Some extractors require models that must be loaded separately.
      Use the load_models() function to initialize all required models.
      
      Heavy extractors (AMR, ontology) that require GPU resources are
      marked as such. Others run on CPU.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class ExtractionResult:
    """Standardized result from any extraction function."""
    extractor: str
    text: str
    features: Any
    metadata: Dict = field(default_factory=dict)


# ============================================================
# Model Loading
# ============================================================

def load_models(extractors: List[str] = None, device: str = "cpu") -> Dict:
    """Load models required for specified extractors.
    
    Args:
        extractors: List of extractor names to load models for.
                   Options: ['ner', 'pos', 'keywords', 'amr', 'ontology']
                   None = load all
        device: 'cpu' or 'cuda'
        
    Returns:
        Dict of loaded models keyed by extractor name
    """
    if extractors is None:
        extractors = ['ner', 'pos', 'keywords']  # Safe defaults (CPU-only)
    
    models = {}
    
    if 'ner' in extractors or 'pos' in extractors:
        import spacy
        models['nlp'] = spacy.load('en_core_web_lg')
    
    if 'keywords' in extractors:
        try:
            from keybert import KeyBERT
            models['keybert'] = KeyBERT()
        except ImportError:
            print("Warning: keybert not installed. Keyword extraction unavailable.")
    
    if 'amr' in extractors:
        try:
            import amrlib
            models['amr_stog'] = amrlib.load_stog_model()
        except (ImportError, Exception) as e:
            print(f"Warning: AMR model not available: {e}")
    
    return models


# ============================================================
# Named Entity Recognition (NER)
# ============================================================

def extract_ner(text: str, models: Dict = None) -> List[List[str]]:
    """Extract named entities using spaCy.
    
    Args:
        text: Input text
        models: Dict with 'nlp' key containing spaCy model
        
    Returns:
        List of [entity_text, entity_label] pairs
    """
    if not text or not isinstance(text, str):
        return []
    
    if models and 'nlp' in models:
        nlp = models['nlp']
    else:
        import spacy
        nlp = spacy.load('en_core_web_lg')
    
    doc = nlp(text)
    return [[ent.text, ent.label_] for ent in doc.ents]


def get_entity_names(entities: List[List[str]]) -> List[str]:
    """Extract just entity names from NER output."""
    return [ent[0] for ent in entities]


# ============================================================
# Part-of-Speech Tagging (POS)
# ============================================================

def extract_pos(text: str, models: Dict = None) -> List[List[str]]:
    """Extract POS tags using spaCy.
    
    Args:
        text: Input text
        models: Dict with 'nlp' key containing spaCy model
        
    Returns:
        List of [token, tag, dep] triples
    """
    if not text or not isinstance(text, str):
        return []
    
    if models and 'nlp' in models:
        nlp = models['nlp']
    else:
        import spacy
        nlp = spacy.load('en_core_web_lg')
    
    doc = nlp(text)
    return [[token.text, token.tag_, token.dep_] for token in doc]


# ============================================================
# Keyword Extraction
# ============================================================

def extract_keywords(text: str, models: Dict = None, 
                     top_n: int = 500) -> List[Tuple[str, float]]:
    """Extract keywords using KeyBERT.
    
    Args:
        text: Input text
        models: Dict with 'keybert' key
        top_n: Maximum number of keywords
        
    Returns:
        List of (keyword, score) tuples
    """
    if not text or not isinstance(text, str):
        return []
    
    if models and 'keybert' in models:
        kw_model = models['keybert']
    else:
        try:
            from keybert import KeyBERT
            kw_model = KeyBERT()
        except ImportError:
            print("keybert not installed")
            return []
    
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    return keywords


# ============================================================
# Abstract Meaning Representation (AMR) [GPU recommended]
# ============================================================

def extract_amr(text: str, models: Dict = None) -> str:
    """Extract AMR graph from text.
    
    Note: Best performance with GPU. CPU is very slow for large texts.
    
    Args:
        text: Input text (single sentence recommended)
        models: Dict with 'amr_stog' key
        
    Returns:
        AMR graph string in PENMAN notation
    """
    if not text or not isinstance(text, str):
        return ""
    
    if models and 'amr_stog' in models:
        stog = models['amr_stog']
    else:
        try:
            import amrlib
            stog = amrlib.load_stog_model()
        except (ImportError, Exception) as e:
            return f"AMR unavailable: {e}"
    
    try:
        graphs = stog.parse_sents([text])
        return graphs[0] if graphs else ""
    except Exception as e:
        return f"AMR error: {e}"


def extract_amr_batch(texts: List[str], models: Dict = None, 
                      batch_size: int = 32) -> List[str]:
    """Batch AMR extraction for efficiency.
    
    Args:
        texts: List of input texts
        models: Dict with 'amr_stog' key
        batch_size: Processing batch size
        
    Returns:
        List of AMR graph strings
    """
    if models and 'amr_stog' in models:
        stog = models['amr_stog']
    else:
        try:
            import amrlib
            stog = amrlib.load_stog_model()
        except (ImportError, Exception) as e:
            return [f"AMR unavailable: {e}"] * len(texts)
    
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Filter out empty/non-string
        valid = [(j, t) for j, t in enumerate(batch) if t and isinstance(t, str)]
        if valid:
            indices, valid_texts = zip(*valid)
            try:
                graphs = stog.parse_sents(list(valid_texts))
            except Exception:
                graphs = [""] * len(valid_texts)
            
            batch_results = [""] * len(batch)
            for idx, graph in zip(indices, graphs):
                batch_results[idx] = graph
            results.extend(batch_results)
        else:
            results.extend([""] * len(batch))
    
    return results


# ============================================================
# First-Order Logic (FOL)
# ============================================================

def extract_fol(text: str, amr_graph: str = None, models: Dict = None) -> str:
    """Generate first-order logic representation.
    
    Can work from raw text or from a pre-computed AMR graph.
    
    Args:
        text: Input text
        amr_graph: Optional pre-computed AMR graph
        models: Dict with required models
        
    Returns:
        FOL representation string
    """
    # FOL generation typically requires AMR as intermediate
    if not amr_graph:
        amr_graph = extract_amr(text, models)
    
    if not amr_graph or amr_graph.startswith("AMR"):
        return ""
    
    # Basic AMR-to-FOL conversion
    # Full implementation would use amr2fol or similar
    try:
        # Placeholder for full FOL conversion
        # In production, this uses the amr-to-fol pipeline
        return f"FOL({amr_graph[:100]}...)"
    except Exception as e:
        return f"FOL error: {e}"


# ============================================================
# Ontology Mapping
# ============================================================

def extract_ontology(text: str, entities: List[List[str]] = None,
                     keywords: List[Tuple[str, float]] = None,
                     ontology_index: Dict = None) -> Dict:
    """Map text entities and keywords to ontology concepts.
    
    Args:
        text: Input text
        entities: Pre-extracted NER entities (optional, will extract if None)
        keywords: Pre-extracted keywords (optional)
        ontology_index: Loaded ontology index (Wikidata/DBpedia)
        
    Returns:
        Dict with mapped ontology concepts
    """
    if entities is None:
        entities = extract_ner(text)
    
    entity_names = get_entity_names(entities)
    
    if ontology_index is None:
        return {
            "entities": entity_names,
            "concepts": [],
            "note": "No ontology index loaded"
        }
    
    # Map entities to ontology
    concepts = []
    for name in entity_names:
        if name.lower() in ontology_index:
            concepts.append(ontology_index[name.lower()])
    
    return {
        "entities": entity_names,
        "concepts": concepts,
    }


# ============================================================
# Convenience: Extract All Features
# ============================================================

def extract_all_symbolic(text: str, models: Dict = None,
                         include_gpu: bool = False) -> Dict[str, Any]:
    """Extract all symbolic features from text.
    
    Args:
        text: Input text
        models: Pre-loaded models
        include_gpu: Whether to include GPU-heavy extractors (AMR)
        
    Returns:
        Dict with all extracted features
    """
    results = {}
    
    results['ner'] = extract_ner(text, models)
    results['pos'] = extract_pos(text, models)
    results['keywords'] = extract_keywords(text, models)
    
    if include_gpu:
        results['amr'] = extract_amr(text, models)
        results['fol'] = extract_fol(text, amr_graph=results.get('amr'), models=models)
    
    return results
