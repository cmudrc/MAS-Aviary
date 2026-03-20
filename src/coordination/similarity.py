"""Similarity computation for iterative feedback metrics.

Provides cosine similarity between agent output texts using TF-IDF
vectorization (default) or Jaccard token overlap (fallback).
"""

from __future__ import annotations


def _tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using TF-IDF vectors.

    Returns a float in [0.0, 1.0]. Returns 1.0 for identical texts,
    0.0 for completely dissimilar texts.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text_a, text_b])
    sim = _cos_sim(tfidf[0:1], tfidf[1:2])[0][0]
    return float(min(max(sim, 0.0), 1.0))


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard similarity between two texts (token overlap).

    Returns |intersection| / |union| of whitespace-split token sets.
    Returns 1.0 for identical token sets, 0.0 for disjoint sets.
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0  # both empty → identical
    if not tokens_a or not tokens_b:
        return 0.0  # one empty, one not → no overlap
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def compute_similarity(text_a: str, text_b: str, method: str = "tfidf") -> float:
    """Compute similarity between two text strings.

    Args:
        text_a: First text.
        text_b: Second text.
        method: "tfidf" (default) or "jaccard".

    Returns:
        Similarity score in [0.0, 1.0].

    Raises:
        ValueError: If method is not recognised.
    """
    # Guard: empty inputs.
    if not text_a.strip() and not text_b.strip():
        return 1.0
    if not text_a.strip() or not text_b.strip():
        return 0.0

    if method == "tfidf":
        try:
            return _tfidf_cosine_similarity(text_a, text_b)
        except ImportError:
            # sklearn not available — fall back to Jaccard.
            return _jaccard_similarity(text_a, text_b)
    elif method == "jaccard":
        return _jaccard_similarity(text_a, text_b)
    else:
        raise ValueError(f"Unknown similarity method {method!r}. Use 'tfidf' or 'jaccard'.")
