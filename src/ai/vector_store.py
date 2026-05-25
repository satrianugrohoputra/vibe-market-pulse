"""
src/ai/vector_store.py
======================
Lightweight ChromaDB wrapper for local, in-memory vector storage.

Design decisions
----------------
* **In-memory only** — `chromadb.Client()` (ephemeral) so nothing is written
  to disk.  Fast, zero setup, resets each session.  Perfect for Streamlit Cloud
  free tier where persistent storage is unreliable.
* **One collection per session** — identified by a hash of the review texts so
  we never re-index the same data twice within a session.
* **L2-normalized embeddings** — produced by embeddings.py, so cosine
  similarity == inner product.  ChromaDB `cosine` space is used explicitly.

Public API
----------
    build_index(reviews, embeddings)    -> ReviewIndex
    ReviewIndex.query(query_vec, k)     -> list[str]   top-k review texts
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import chromadb
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_reviews(reviews: list[str]) -> str:
    """Return a short SHA-256 hex digest of the joined review list."""
    joined = "\n".join(reviews)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ReviewIndex — the public data structure
# ---------------------------------------------------------------------------

@dataclass
class ReviewIndex:
    """
    Holds a ChromaDB in-memory collection and exposes a simple query interface.

    Attributes
    ----------
    collection : chromadb.Collection
        The underlying Chroma collection (cosine distance space).
    review_hash : str
        SHA-256 digest of the indexed reviews — used to detect stale indices.
    n_docs : int
        Number of documents stored in the index.
    """
    collection: chromadb.Collection
    review_hash: str
    n_docs: int = 0
    _raw_reviews: list[str] = field(default_factory=list, repr=False)

    def query(
        self,
        query_vector: np.ndarray,
        k: int = 20,
    ) -> list[str]:
        """
        Retrieve the top-k most semantically similar reviews.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (384,) — L2-normalized query embedding from embed_query().
        k : int
            Number of results to return.  Capped at n_docs automatically.

        Returns
        -------
        list[str]
            Review texts ranked by cosine similarity (highest first).
        """
        k_safe = min(k, self.n_docs)
        if k_safe == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=k_safe,
            include=["documents"],
        )
        docs: list[str] = results.get("documents", [[]])[0]
        return docs


# ---------------------------------------------------------------------------
# Module-level cache: keyed by review_hash so we never re-embed same data
# ---------------------------------------------------------------------------
_INDEX_CACHE: dict[str, ReviewIndex] = {}


def build_index(
    reviews: list[str],
    embeddings: np.ndarray,
    *,
    force_rebuild: bool = False,
) -> ReviewIndex:
    """
    Build (or retrieve from cache) a ChromaDB in-memory index.

    Parameters
    ----------
    reviews : list[str]
        Raw review texts to index.  Should be filtered negative reviews only.
    embeddings : np.ndarray
        Pre-computed embeddings from embed_texts(), shape (len(reviews), 384).
    force_rebuild : bool
        If True, ignore the cache and rebuild from scratch.

    Returns
    -------
    ReviewIndex
        Ready-to-query index object.

    Notes
    -----
    * Empty / whitespace-only reviews are skipped silently.
    * ChromaDB requires unique string IDs per document.  We use zero-padded
      integer indices ("doc_00000", "doc_00001", …).
    """
    global _INDEX_CACHE

    # Filter out blank reviews (keep index aligned with embeddings)
    valid_pairs = [
        (text, vec)
        for text, vec in zip(reviews, embeddings)
        if str(text).strip()
    ]

    if not valid_pairs:
        # Return an empty stub index so callers don't have to guard for None
        client = chromadb.Client()
        stub_collection = client.create_collection(
            name="stub_empty",
            metadata={"hnsw:space": "cosine"},
        )
        return ReviewIndex(
            collection=stub_collection,
            review_hash="empty",
            n_docs=0,
            _raw_reviews=[],
        )

    texts, vecs = zip(*valid_pairs)
    texts = list(texts)
    vecs_array = np.array(vecs, dtype=np.float32)

    review_hash = _hash_reviews(texts)

    # Return cached index if hash matches and rebuild not forced
    if not force_rebuild and review_hash in _INDEX_CACHE:
        return _INDEX_CACHE[review_hash]

    # --- Build fresh ChromaDB in-memory collection ---
    client = chromadb.Client()                      # ephemeral (in-memory)
    collection_name = f"reviews_{review_hash}"

    # Create with cosine distance metric
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch-upsert documents
    ids = [f"doc_{i:05d}" for i in range(len(texts))]
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=vecs_array.tolist(),
    )

    index = ReviewIndex(
        collection=collection,
        review_hash=review_hash,
        n_docs=len(texts),
        _raw_reviews=texts,
    )
    _INDEX_CACHE[review_hash] = index
    return index


def clear_cache() -> None:
    """Wipe the module-level index cache (useful for testing)."""
    global _INDEX_CACHE
    _INDEX_CACHE.clear()
