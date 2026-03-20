"""Tests for similarity computation (TF-IDF and Jaccard).

No GPU needed. Tests cosine and Jaccard similarity between text pairs.
"""

from unittest.mock import patch

import pytest

from src.coordination.similarity import (
    _jaccard_similarity,
    _tfidf_cosine_similarity,
    compute_similarity,
)

# ---- TF-IDF cosine similarity -----------------------------------------------


class TestTfidfSimilarity:
    def test_identical_texts(self):
        sim = _tfidf_cosine_similarity("hello world foo", "hello world foo")
        assert sim == pytest.approx(1.0)

    def test_completely_different_texts(self):
        sim = _tfidf_cosine_similarity("alpha beta gamma", "delta epsilon zeta")
        assert sim == pytest.approx(0.0, abs=0.01)

    def test_partially_similar_texts(self):
        sim = _tfidf_cosine_similarity(
            "the cat sat on the mat",
            "the cat sat on the floor",
        )
        assert 0.3 < sim < 1.0

    def test_order_independent(self):
        a = "aircraft parameters setup"
        b = "setup aircraft parameters"
        sim_ab = _tfidf_cosine_similarity(a, b)
        sim_ba = _tfidf_cosine_similarity(b, a)
        assert sim_ab == pytest.approx(sim_ba)


# ---- Jaccard similarity -----------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_token_sets(self):
        assert _jaccard_similarity("a b c", "a b c") == pytest.approx(1.0)

    def test_disjoint_token_sets(self):
        assert _jaccard_similarity("a b c", "d e f") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # intersection={a,b}, union={a,b,c,d} → 2/4 = 0.5
        assert _jaccard_similarity("a b c", "a b d") == pytest.approx(0.5)

    def test_both_empty(self):
        assert _jaccard_similarity("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert _jaccard_similarity("hello", "") == pytest.approx(0.0)
        assert _jaccard_similarity("", "hello") == pytest.approx(0.0)

    def test_case_insensitive(self):
        assert _jaccard_similarity("Hello World", "hello world") == pytest.approx(1.0)


# ---- compute_similarity dispatch --------------------------------------------


class TestComputeSimilarity:
    def test_tfidf_method(self):
        sim = compute_similarity("hello world", "hello world", method="tfidf")
        assert sim == pytest.approx(1.0)

    def test_jaccard_method(self):
        sim = compute_similarity("a b c", "a b d", method="jaccard")
        assert sim == pytest.approx(0.5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown similarity method"):
            compute_similarity("a", "b", method="magic")

    def test_both_empty_strings(self):
        assert compute_similarity("", "", method="tfidf") == pytest.approx(1.0)
        assert compute_similarity("  ", "  ", method="jaccard") == pytest.approx(1.0)

    def test_one_empty_string(self):
        assert compute_similarity("hello", "", method="tfidf") == pytest.approx(0.0)
        assert compute_similarity("", "hello", method="jaccard") == pytest.approx(0.0)

    def test_default_method_is_tfidf(self):
        sim = compute_similarity("test text", "test text")
        assert sim == pytest.approx(1.0)

    def test_tfidf_fallback_to_jaccard_on_import_error(self):
        """If sklearn is unavailable, TF-IDF falls back to Jaccard."""
        with patch(
            "src.coordination.similarity._tfidf_cosine_similarity",
            side_effect=ImportError("no sklearn"),
        ):
            sim = compute_similarity("a b c", "a b c", method="tfidf")
            assert sim == pytest.approx(1.0)


# ---- Similarity properties ---------------------------------------------------


class TestSimilarityProperties:
    def test_similarity_is_between_0_and_1(self):
        pairs = [
            ("hello world", "hello world"),
            ("foo bar", "baz qux"),
            ("partial match here", "partial overlap here"),
        ]
        for a, b in pairs:
            for method in ("tfidf", "jaccard"):
                sim = compute_similarity(a, b, method=method)
                assert 0.0 <= sim <= 1.0, f"Out of range for {method}: {sim}"

    def test_similarity_is_symmetric(self):
        a = "the quick brown fox"
        b = "a slow red fox"
        for method in ("tfidf", "jaccard"):
            sim_ab = compute_similarity(a, b, method=method)
            sim_ba = compute_similarity(b, a, method=method)
            assert sim_ab == pytest.approx(sim_ba, abs=1e-9)
