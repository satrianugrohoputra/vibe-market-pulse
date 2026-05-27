"""
Aspect Clustering Module — Phase 3
==================================
Lightweight clustering of retrieved review texts into "aspect groups" using
TF-IDF + KMeans. NO LLM CALLS — pure scikit-learn, runs in milliseconds.

Why this module exists:
- Without clustering, Gemini may focus too heavily on the loudest complaint
  type and miss other aspects.
- With clustering, we identify N distinct "complaint themes" and feed Gemini
  a balanced sample (round-robin from each cluster), forcing broader coverage.
- The cluster keywords also drive the rule-based validate_report check —
  no LLM needed to verify aspect coverage.

Public API:
    cluster_reviews(texts, n_clusters=4) -> list[ClusterInfo]
    extract_top_keywords(vectorizer, model, n=5) -> list[list[str]]

Each ClusterInfo dict:
    {
        "cluster_id": int,
        "size": int,
        "keywords": list[str],          # top-N TF-IDF terms for this cluster
        "sample_texts": list[str],      # representative reviews (closest to centroid)
        "all_indices": list[int],       # indices into the input `texts` list
    }
"""

from __future__ import annotations

from typing import TypedDict


class ClusterInfo(TypedDict):
    cluster_id: int
    size: int
    keywords: list[str]
    sample_texts: list[str]
    all_indices: list[int]


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_N_CLUSTERS = 4
TOP_KEYWORDS_PER_CLUSTER = 5
SAMPLES_PER_CLUSTER = 3
MIN_TEXTS_FOR_CLUSTERING = 6  # below this we skip clustering


# ----------------------------------------------------------------------------
# Core clustering function
# ----------------------------------------------------------------------------
def cluster_reviews(
    texts: list[str],
    n_clusters: int = DEFAULT_N_CLUSTERS,
) -> list[ClusterInfo]:
    """
    Group `texts` into `n_clusters` clusters using TF-IDF + KMeans.

    Returns:
        Empty list if input is too small to cluster meaningfully (<6 texts).
        Otherwise a list of ClusterInfo dicts sorted by cluster size (desc).

    Behaviour notes:
    - Auto-shrinks n_clusters if there are fewer texts than requested.
    - Uses English stopwords by default — Indonesian texts still cluster well
      because rare keywords (negative ones) carry the signal.
    - Falls back to "single-cluster" if KMeans fails (e.g. all-identical text).
    """
    # Filter empty / whitespace-only strings (preserve original indices)
    valid_pairs = [
        (i, t.strip()) for i, t in enumerate(texts) if t and t.strip()
    ]
    if len(valid_pairs) < MIN_TEXTS_FOR_CLUSTERING:
        return []

    valid_indices = [p[0] for p in valid_pairs]
    valid_texts = [p[1] for p in valid_pairs]

    # Auto-shrink k if we have fewer docs than requested clusters
    k = max(2, min(n_clusters, len(valid_texts) // 2))

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import KMeans
        import numpy as np

        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        X = vectorizer.fit_transform(valid_texts)

        # Edge case: vocabulary too small
        if X.shape[1] < 2:
            return []

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            max_iter=200,
        )
        labels = model.fit_predict(X)

        # Top keywords per cluster (closest to centroid in TF-IDF space)
        feature_names = list(vectorizer.get_feature_names_out())
        centroids = model.cluster_centers_

        clusters: list[ClusterInfo] = []
        for cid in range(k):
            member_idx = np.where(labels == cid)[0].tolist()
            if not member_idx:
                continue

            # Top-N keywords for this cluster centroid
            top_idx = centroids[cid].argsort()[::-1][:TOP_KEYWORDS_PER_CLUSTER]
            keywords = [feature_names[j] for j in top_idx]

            # Representative samples: members closest to centroid
            from sklearn.metrics.pairwise import cosine_similarity
            member_vecs = X[member_idx]
            sims = cosine_similarity(member_vecs, centroids[cid].reshape(1, -1))
            ranked = sorted(
                zip(member_idx, sims.flatten().tolist()),
                key=lambda p: p[1],
                reverse=True,
            )[:SAMPLES_PER_CLUSTER]
            sample_texts = [valid_texts[i] for i, _ in ranked]

            # Map back to ORIGINAL input indices
            original_indices = [valid_indices[i] for i in member_idx]

            clusters.append({
                "cluster_id": int(cid),
                "size": len(member_idx),
                "keywords": keywords,
                "sample_texts": sample_texts,
                "all_indices": original_indices,
            })

        # Sort by size, largest cluster first
        clusters.sort(key=lambda c: c["size"], reverse=True)
        return clusters

    except Exception:
        # Any clustering failure → caller can fall back to no clustering
        return []


# ----------------------------------------------------------------------------
# Round-robin balanced sampling across clusters
# ----------------------------------------------------------------------------
def balanced_sample_from_clusters(
    clusters: list[ClusterInfo],
    texts: list[str],
    target_size: int,
) -> list[str]:
    """
    Pick `target_size` texts by round-robin sampling across clusters so every
    aspect is represented (instead of letting one big cluster dominate).

    Falls back to an empty list if clusters is empty.
    """
    if not clusters or target_size <= 0:
        return []

    selected: list[str] = []
    pointers = [0] * len(clusters)

    # Round-robin: take 1 from cluster 0, 1 from cluster 1, ... and repeat
    while len(selected) < target_size:
        progress = False
        for ci, cluster in enumerate(clusters):
            if pointers[ci] < len(cluster["all_indices"]):
                idx = cluster["all_indices"][pointers[ci]]
                if 0 <= idx < len(texts):
                    selected.append(texts[idx])
                pointers[ci] += 1
                progress = True
                if len(selected) >= target_size:
                    break
        if not progress:
            break  # all clusters exhausted

    return selected


# ----------------------------------------------------------------------------
# Validation: does a report mention each cluster's keywords?
# ----------------------------------------------------------------------------
def validate_report_coverage(
    report_text: str,
    clusters: list[ClusterInfo],
    min_keywords_per_cluster: int = 1,
) -> dict:
    """
    Rule-based check: for each cluster, count how many of its top keywords
    appear in the report. A cluster is "covered" if at least
    `min_keywords_per_cluster` of its keywords are present.

    Returns:
        {
            "is_valid": bool,                         # all clusters covered
            "coverage_pct": float,                    # 0..1
            "covered_clusters": list[int],            # cluster IDs covered
            "missed_clusters": list[ClusterInfo],     # clusters NOT covered
        }
    """
    if not clusters or not report_text:
        return {
            "is_valid": True,
            "coverage_pct": 1.0,
            "covered_clusters": [],
            "missed_clusters": [],
        }

    report_lower = report_text.lower()
    covered: list[int] = []
    missed: list[ClusterInfo] = []

    for cluster in clusters:
        # Count how many of this cluster's keywords appear in the report
        hits = sum(
            1 for kw in cluster["keywords"]
            if kw.lower() in report_lower
        )
        if hits >= min_keywords_per_cluster:
            covered.append(cluster["cluster_id"])
        else:
            missed.append(cluster)

    coverage = len(covered) / len(clusters) if clusters else 1.0
    return {
        "is_valid": len(missed) == 0,
        "coverage_pct": coverage,
        "covered_clusters": covered,
        "missed_clusters": missed,
    }
