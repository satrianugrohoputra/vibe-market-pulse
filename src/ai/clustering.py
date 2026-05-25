"""
src/ai/clustering.py
====================
KMeans-based dynamic clustering of retrieved negative reviews.

Why KMeans (not DBSCAN)?
------------------------
* KMeans always produces exactly k clusters → predictable UI output.
* For 15-20 retrieved reviews, DBSCAN's density estimation is unreliable.
* DBSCAN can label everything as noise (-1) when data is sparse.

Auto-k selection
----------------
If k is not specified, we use the **silhouette score** to pick the best k
in the range [2, 5].  For fewer than 6 reviews we fall back to k=2.

Cluster labeling
----------------
Labels are derived from per-cluster **TF-IDF top terms** — no Gemini call
needed here.  Each cluster gets a human-readable name like:
    "Cluster 1 · sizing, fit, small, large, tight"

Public API
----------
    cluster_reviews(texts, embeddings, k)  -> list[ClusterResult]
    ClusterResult                           (dataclass)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    """
    One cluster of thematically similar negative reviews.

    Attributes
    ----------
    cluster_id : int
        Zero-based cluster index.
    label : str
        Human-readable label derived from top TF-IDF terms.
        e.g. "Cluster 1 · sizing, fit, small, large, tight"
    reviews : list[str]
        Review texts assigned to this cluster.
    top_terms : list[str]
        Top-5 TF-IDF keywords driving this cluster's theme.
    size : int
        Number of reviews in this cluster.
    """
    cluster_id: int
    label: str
    reviews: list[str]
    top_terms: list[str] = field(default_factory=list)
    size: int = 0

    def __post_init__(self) -> None:
        self.size = len(self.reviews)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_k(embeddings: np.ndarray, k_range: range) -> int:
    """
    Select the best k using silhouette score.

    Falls back to 2 if silhouette cannot be computed (e.g. too few samples).
    """
    best_k = k_range.start
    best_score = -1.0

    for k in k_range:
        if k >= len(embeddings):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels, metric="cosine")
        except ValueError:
            continue
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _extract_top_terms(texts: list[str], n_terms: int = 5) -> list[str]:
    """
    Extract the top n TF-IDF terms from a small list of texts.

    Returns a list of lowercase single words (stopwords removed).
    Falls back to empty list if texts are too sparse.
    """
    if not texts:
        return []

    # Minimal stop word list (EN + ID) to keep dependency-free
    _STOPWORDS = {
        # English
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "it", "this", "that", "was", "are", "be", "as",
        "i", "my", "me", "we", "you", "he", "she", "they", "have", "had",
        "not", "so", "do", "did", "will", "just", "get", "got", "been",
        "would", "could", "should", "very", "also", "if", "from", "up",
        "about", "out", "no", "its", "than", "then", "there", "when",
        # Indonesian
        "yang", "dan", "di", "ke", "dari", "ini", "itu", "ada", "tidak",
        "juga", "sudah", "dengan", "untuk", "lebih", "tapi", "karena",
        "sangat", "saya", "iya", "ya", "nya", "pun", "aja", "banget",
        "kalau", "bisa", "buat", "pas", "lagi", "sama", "karna", "udah",
    }

    try:
        tfidf = TfidfVectorizer(
            max_features=200,
            ngram_range=(1, 1),
            min_df=1,
            token_pattern=r"\b[a-zA-Z]{3,}\b",   # at least 3 chars
        )
        matrix = tfidf.fit_transform(texts)
        feature_names = tfidf.get_feature_names_out()
        mean_scores = matrix.mean(axis=0).A1  # type: ignore[union-attr]
        ranked = np.argsort(mean_scores)[::-1]

        terms: list[str] = []
        for idx in ranked:
            term = feature_names[idx].lower()
            if term not in _STOPWORDS and len(terms) < n_terms:
                terms.append(term)
            if len(terms) == n_terms:
                break
        return terms

    except Exception:
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_reviews(
    texts: list[str],
    embeddings: np.ndarray,
    k: int | None = None,
) -> list[ClusterResult]:
    """
    Group retrieved negative reviews into thematic clusters.

    Parameters
    ----------
    texts : list[str]
        Review texts (same order as `embeddings`).
    embeddings : np.ndarray
        Shape (N, 384) — L2-normalized embeddings from embed_texts().
    k : int | None
        Number of clusters.  If None, auto-selected via silhouette score in
        the range [2, 5] (or [2, min(5, N-1)] for small N).

    Returns
    -------
    list[ClusterResult]
        Sorted by cluster_id, smallest first.  Empty list if fewer than 2
        reviews are passed.

    Notes
    -----
    * Clusters with 0 reviews are omitted from the output.
    * Embeddings are re-normalized before KMeans to ensure cosine geometry.
    """
    if len(texts) < 2:
        # Not enough data to cluster — return one "cluster" with everything
        if texts:
            top = _extract_top_terms(texts)
            label = "Cluster 1 · " + ", ".join(top) if top else "Cluster 1"
            return [ClusterResult(cluster_id=0, label=label, reviews=texts, top_terms=top)]
        return []

    # Re-normalize embeddings (belt-and-suspenders)
    normed = normalize(embeddings, norm="l2")

    # Auto-pick k if not provided
    if k is None:
        max_k = min(5, len(texts) - 1)
        if max_k < 2:
            k = 1
        else:
            k = _pick_k(normed, range(2, max_k + 1))

    k = max(1, min(k, len(texts)))  # clamp: 1 ≤ k ≤ N

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels_array = km.fit_predict(normed)

    # Group texts by cluster label
    cluster_map: dict[int, list[str]] = {i: [] for i in range(k)}
    for text, label_id in zip(texts, labels_array):
        cluster_map[int(label_id)].append(text)

    # Build ClusterResult objects with TF-IDF labels
    results: list[ClusterResult] = []
    for cid in range(k):
        cluster_texts = cluster_map[cid]
        if not cluster_texts:
            continue  # skip empty clusters
        top_terms = _extract_top_terms(cluster_texts)
        label = f"Cluster {cid + 1}"
        if top_terms:
            label += " · " + ", ".join(top_terms)
        results.append(
            ClusterResult(
                cluster_id=cid,
                label=label,
                reviews=cluster_texts,
                top_terms=top_terms,
            )
        )

    # Sort by size descending (largest complaint cluster first)
    results.sort(key=lambda c: c.size, reverse=True)

    return results
