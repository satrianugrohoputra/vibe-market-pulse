"""
Vector Store Module — NumPy In-Memory Backend
==============================================
Encapsulates embedding generation and semantic retrieval using
sentence-transformers + a pure-NumPy in-memory index.

Why NOT ChromaDB:
- ChromaDB's EphemeralClient stores state in a module-level SQLite
  singleton. Streamlit reruns the script on every interaction, which
  can wipe ChromaDB's tenant/collection state even though our Python
  wrapper object survives in st.session_state. Symptoms in production:
    * "Could not connect to tenant default_tenant"
    * sqlite3.OperationalError on .query() after successful build
- A pure NumPy index is simpler, faster for in-memory datasets up to
  ~50k rows, and *truly* survives Streamlit reruns because numpy arrays
  serialize cleanly into st.session_state.

Design choices:
- Multilingual embedding model (paraphrase-multilingual-MiniLM-L12-v2)
  → handles both English and Indonesian datasets transparently.
- Embeddings stored as L2-normalized float32 numpy arrays so cosine
  similarity reduces to a single matmul (very fast).
- DataFrame-hash caching via st.session_state → avoids re-embedding when
  the user re-renders the same uploaded file.

Public API (UNCHANGED — drop-in replacement):
    get_or_build_vector_store(df, text_col, ...) -> (store, n_indexed)
    ReviewVectorStore.search(query, top_k, sentiment_filter) -> list[dict]
    ReviewVectorStore.search_negatives(topic, top_k) -> list[dict]
    ReviewVectorStore.indexed_count -> int
"""

from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 64

# Maximum number of reviews to index. Beyond this, sample randomly to keep
# upload-flow snappy. 30k × 384 dims × 4 bytes ≈ 46 MB — fits easily.
MAX_INDEX_SIZE = 30000


# ----------------------------------------------------------------------------
# Cached singletons
# ----------------------------------------------------------------------------
@st.cache_resource(
    show_spinner="Loading multilingual embedding model (one-time, ~120MB)..."
)
def get_embedding_model():
    """Load and cache the sentence-transformers model across reruns."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def compute_df_hash(df: pd.DataFrame, text_col: str) -> str:
    """
    Compute a stable hash for a DataFrame based on its text column content.
    Used to detect if the user re-uploaded the same file and skip re-indexing.
    """
    sample_texts = df[text_col].fillna("").astype(str).tolist()[:100]
    text_concat = "||".join(sample_texts)
    fingerprint = f"{text_col}:{len(df)}:{text_concat}"
    return hashlib.md5(fingerprint.encode("utf-8")).hexdigest()


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize so cosine similarity = dot product."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)  # avoid div-by-zero
    return (matrix / norms).astype(np.float32)


# ----------------------------------------------------------------------------
# ReviewVectorStore — pure NumPy in-memory cosine retrieval
# ----------------------------------------------------------------------------
class ReviewVectorStore:
    """
    In-memory vector store using NumPy for cosine similarity search.

    Internal state (everything lives in plain Python / numpy — no SQLite):
        _embeddings: np.ndarray, shape (N, D), float32, L2-normalized
        _documents:  list[str]                              len N
        _metadatas:  list[dict]                             len N
        _ids:        list[str]                              len N
    """

    def __init__(self):
        self.embedder = get_embedding_model()

        # Storage — populated by index_dataframe()
        self._embeddings: Optional[np.ndarray] = None
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._ids: list[str] = []

        self.text_col: Optional[str] = None
        self.indexed_count: int = 0

    # --------------------------------------------------------------------
    # Indexing
    # --------------------------------------------------------------------
    def index_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        sentiment_col: Optional[str] = None,
        rating_col: Optional[str] = None,
        progress_callback=None,
    ) -> int:
        """
        Embed and index all valid rows of the dataframe into the in-memory store.

        Args:
            df: source dataframe (after prediction).
            text_col: column with review text.
            sentiment_col: optional column for "Positive"/"Negative" filter.
            rating_col: optional rating column for numeric filter.
            progress_callback: fn(done, total) called per batch for UI updates.

        Returns:
            Number of rows successfully indexed.
        """
        # Reset state for a fresh build
        self._embeddings = None
        self._documents = []
        self._metadatas = []
        self._ids = []
        self.text_col = text_col
        self.indexed_count = 0

        # Filter: drop empty text rows
        valid_mask = df[text_col].fillna("").astype(str).str.strip().astype(bool)
        valid_df = df.loc[valid_mask].reset_index(drop=True)

        if len(valid_df) == 0:
            return 0

        # Sample if dataset is huge (preserves diversity, keeps UX snappy)
        if len(valid_df) > MAX_INDEX_SIZE:
            valid_df = valid_df.sample(
                n=MAX_INDEX_SIZE, random_state=42
            ).reset_index(drop=True)

        texts = valid_df[text_col].astype(str).tolist()
        ids = [f"row_{i}" for i in range(len(valid_df))]

        # Build metadata (must be primitives: str/int/float/bool)
        metadatas: list[dict] = []
        for i in range(len(valid_df)):
            meta: dict = {"row_idx": i}
            if sentiment_col and sentiment_col in valid_df.columns:
                meta["sentiment"] = str(valid_df.iloc[i][sentiment_col])
            if rating_col and rating_col in valid_df.columns:
                try:
                    meta["rating"] = float(valid_df.iloc[i][rating_col])
                except (ValueError, TypeError):
                    pass
            metadatas.append(meta)

        # Embed in batches with progress callback
        total = len(texts)
        all_embeddings: list[np.ndarray] = []

        for start in range(0, total, EMBED_BATCH_SIZE):
            end = min(start + EMBED_BATCH_SIZE, total)
            batch_texts = texts[start:end]

            batch_emb = self.embedder.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_embeddings.append(batch_emb)

            if progress_callback is not None:
                progress_callback(end, total)

        # Stack and L2-normalize once at the end (cosine = dot product after norm)
        stacked = np.vstack(all_embeddings).astype(np.float32)
        self._embeddings = _l2_normalize(stacked)

        self._documents = texts
        self._metadatas = metadatas
        self._ids = ids
        self.indexed_count = total
        return total

    # --------------------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Run cosine semantic search.

        Args:
            query: natural-language query.
            top_k: number of results to return.
            sentiment_filter: e.g. "Positive" or "Negative" to filter
                              by metadata before ranking.

        Returns:
            List of dicts: {id, text, metadata, score} ordered by relevance.
            Score is cosine similarity in [0..1] (1 = perfect match).
        """
        if self._embeddings is None or self.indexed_count == 0:
            return []
        if not query or not query.strip():
            return []

        # Embed query and L2-normalize
        query_vec = self.embedder.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)
        query_vec = _l2_normalize(query_vec)[0]  # shape (D,)

        # Cosine similarity = dot product (since both sides are L2-normalized)
        # scores: shape (N,) in [-1, 1]
        scores = self._embeddings @ query_vec

        # Apply metadata filter (sentiment) BEFORE top-k
        if sentiment_filter:
            mask = np.array(
                [
                    m.get("sentiment") == sentiment_filter
                    for m in self._metadatas
                ],
                dtype=bool,
            )
            if not mask.any():
                # No rows match the filter — fall back to unfiltered search
                # so the UX still shows something useful.
                mask = None
        else:
            mask = None

        if mask is not None:
            # Mask out non-matching rows by setting their score to -inf
            filtered_scores = np.where(mask, scores, -np.inf)
        else:
            filtered_scores = scores

        # Top-K via argpartition (faster than full argsort for large N)
        effective_k = min(top_k, self.indexed_count)
        if effective_k <= 0:
            return []

        # argpartition gives the top-K indices unsorted; we then sort just those.
        if effective_k < len(filtered_scores):
            top_idx = np.argpartition(-filtered_scores, effective_k - 1)[
                :effective_k
            ]
        else:
            top_idx = np.arange(len(filtered_scores))

        # Sort the top-K indices by score descending
        top_idx = top_idx[np.argsort(-filtered_scores[top_idx])]

        # Build result list — drop any -inf (filtered-out) entries
        out: list[dict] = []
        for idx in top_idx:
            score = float(filtered_scores[idx])
            if not np.isfinite(score):
                break  # remaining ones are all filtered out
            # Clamp negative cosine to 0 for nicer UI display (rare edge case)
            score_clamped = max(0.0, min(1.0, score))
            out.append({
                "id": self._ids[idx],
                "text": self._documents[idx],
                "metadata": dict(self._metadatas[idx]),  # defensive copy
                "score": score_clamped,
            })

        return out

    # --------------------------------------------------------------------
    # Convenience: get top-K negative reviews about a topic (for RAG)
    # --------------------------------------------------------------------
    def search_negatives(self, topic: str, top_k: int = 10) -> list[dict]:
        """Targeted retrieval for the Gemini Agentic RAG workflow."""
        if not topic or not topic.strip():
            topic = "negative complaint problem issue bad disappointed"
        return self.search(
            query=topic, top_k=top_k, sentiment_filter="Negative"
        )


# ----------------------------------------------------------------------------
# Cached factory — keyed on dataframe hash
# ----------------------------------------------------------------------------
def get_or_build_vector_store(
    df: pd.DataFrame,
    text_col: str,
    sentiment_col: Optional[str] = None,
    rating_col: Optional[str] = None,
    progress_callback=None,
) -> tuple[ReviewVectorStore, int]:
    """
    Return a vector store for this dataframe, building it only if the
    underlying data changed (detected via hash). Cached in st.session_state.

    Returns:
        (store, n_indexed)
    """
    df_hash = compute_df_hash(df, text_col)
    cache_key = f"_vstore_{df_hash}"

    # Reuse existing store if data is unchanged AND it's still healthy
    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        if (
            isinstance(cached, ReviewVectorStore)
            and cached.indexed_count > 0
            and cached._embeddings is not None
        ):
            return cached, cached.indexed_count
        # Otherwise fall through and rebuild

    store = ReviewVectorStore()
    n = store.index_dataframe(
        df,
        text_col,
        sentiment_col=sentiment_col,
        rating_col=rating_col,
        progress_callback=progress_callback,
    )
    st.session_state[cache_key] = store
    return store, n
