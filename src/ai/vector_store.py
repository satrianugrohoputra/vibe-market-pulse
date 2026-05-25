"""
Vector Store Module
===================
Encapsulates embedding generation and semantic retrieval using
sentence-transformers + ChromaDB (in-memory).

Design choices:
- In-memory ChromaDB → avoids SQLite version issues, fits Streamlit's
  session model, and is fast enough for tens of thousands of reviews.
- Multilingual embedding model (paraphrase-multilingual-MiniLM-L12-v2)
  → handles both English and Indonesian datasets transparently.
- DataFrame-hash caching via st.session_state → avoids re-embedding when
  the user re-renders the same uploaded file.

Public API:
    get_or_build_vector_store(df, text_col, ...) -> (store, n_indexed)
    ReviewVectorStore.search(query, top_k, sentiment_filter) -> list[dict]
"""

from __future__ import annotations

import hashlib
from typing import Optional

import pandas as pd
import streamlit as st


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "reviews"
EMBED_BATCH_SIZE = 64

# Maximum number of reviews to index. Beyond this, sample randomly to keep
# upload-flow snappy. Adjust as needed for production datasets.
MAX_INDEX_SIZE = 30000


# ----------------------------------------------------------------------------
# Cached singletons
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading multilingual embedding model (one-time, ~120MB)...")
def get_embedding_model():
    """Load and cache the sentence-transformers model."""
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


# ----------------------------------------------------------------------------
# ReviewVectorStore — wraps ChromaDB collection + embedder
# ----------------------------------------------------------------------------
class ReviewVectorStore:
    """In-memory ChromaDB vector store for review retrieval."""

    def __init__(self):
        import chromadb

        # In-memory client — avoids SQLite >=3.35 requirement issues
        # and keeps things stateless across Streamlit reruns.
        self.client = chromadb.EphemeralClient()
        self.embedder = get_embedding_model()
        self.collection = None
        self.text_col: Optional[str] = None
        self.indexed_count = 0

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
        Embed and index all valid rows of the dataframe into ChromaDB.

        Args:
            df: source dataframe (after prediction).
            text_col: column with review text.
            sentiment_col: optional column for "Positive"/"Negative" filter.
            rating_col: optional rating column for numeric filter.
            progress_callback: fn(done, total) called per batch for UI updates.

        Returns:
            Number of rows successfully indexed.
        """
        # Drop existing collection (fresh build per upload)
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self.text_col = text_col

        # Filter: drop empty text rows
        valid_mask = df[text_col].fillna("").astype(str).str.strip().astype(bool)
        valid_df = df.loc[valid_mask].reset_index(drop=True)

        if len(valid_df) == 0:
            self.indexed_count = 0
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

        # Embed in batches and add to collection
        total = len(texts)
        for start in range(0, total, EMBED_BATCH_SIZE):
            end = min(start + EMBED_BATCH_SIZE, total)
            batch_texts = texts[start:end]

            embeddings = self.embedder.encode(
                batch_texts,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()

            self.collection.add(
                ids=ids[start:end],
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=metadatas[start:end],
            )

            if progress_callback is not None:
                progress_callback(end, total)

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
            sentiment_filter: e.g. "Positive" or "Negative" to filter by metadata.

        Returns:
            List of dicts: {id, text, metadata, score} ordered by relevance.
        """
        if self.collection is None or self.indexed_count == 0:
            return []

        if not query or not query.strip():
            return []

        query_emb = self.embedder.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).tolist()

        where_clause = None
        if sentiment_filter:
            where_clause = {"sentiment": sentiment_filter}

        # Cap top_k to indexed_count to avoid Chroma errors
        effective_k = min(top_k, self.indexed_count)

        try:
            results = self.collection.query(
                query_embeddings=query_emb,
                n_results=effective_k,
                where=where_clause,
            )
        except Exception:
            # Fallback: search without filter if filter fails
            results = self.collection.query(
                query_embeddings=query_emb,
                n_results=effective_k,
            )

        # Format
        out: list[dict] = []
        if not results or not results.get("ids"):
            return out

        ids_list = results["ids"][0] if results["ids"] else []
        docs_list = results["documents"][0] if results.get("documents") else []
        meta_list = results["metadatas"][0] if results.get("metadatas") else []
        dist_list = results["distances"][0] if results.get("distances") else []

        for i in range(len(ids_list)):
            out.append({
                "id": ids_list[i],
                "text": docs_list[i] if i < len(docs_list) else "",
                "metadata": meta_list[i] if i < len(meta_list) else {},
                # Cosine distance in Chroma is 1 - cosine_similarity, so we
                # convert back to similarity score (0..1 where 1 = perfect).
                "score": (
                    max(0.0, 1.0 - float(dist_list[i]))
                    if i < len(dist_list)
                    else 0.0
                ),
            })

        return out

    # --------------------------------------------------------------------
    # Convenience: get top-K negative reviews about a topic (for RAG)
    # --------------------------------------------------------------------
    def search_negatives(self, topic: str, top_k: int = 10) -> list[dict]:
        """Targeted retrieval for the Gemini RAG workflow."""
        if not topic or not topic.strip():
            # Generic negative-sentiment query
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

    if cache_key in st.session_state:
        cached = st.session_state[cache_key]
        return cached, cached.indexed_count

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
