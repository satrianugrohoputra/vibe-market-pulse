"""
src/ai/embeddings.py
====================
Local embedding layer using HuggingFace sentence-transformers.

Model  : paraphrase-multilingual-MiniLM-L12-v2  (~120 MB)
Support: English + Indonesian (and 50+ other languages) — no Gemini call needed.

The SentenceTransformer instance is cached with Streamlit's @st.cache_resource
so it is loaded ONCE per session and reused across all RAG queries.

Public API
----------
    get_encoder()                       -> SentenceTransformer (cached)
    embed_texts(texts)                  -> np.ndarray  shape (N, 384)
    embed_query(query)                  -> np.ndarray  shape (384,)
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# HuggingFace model ID — multilingual, ~120 MB, supports EN + ID
_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"


@st.cache_resource(show_spinner="🔄 Loading local embedding model (first time only)...")
def get_encoder() -> SentenceTransformer:
    """
    Load and cache the multilingual MiniLM encoder.

    Called once per Streamlit session. Subsequent calls return the cached
    instance immediately without re-downloading or re-loading the model.

    Returns
    -------
    SentenceTransformer
        Ready-to-use encoder that maps text → 384-dim float32 vectors.
    """
    return SentenceTransformer(_MODEL_ID)


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Encode a list of review texts into embedding vectors.

    Parameters
    ----------
    texts : list[str]
        Raw review strings (EN or ID). Empty strings are safe — the model
        returns a near-zero vector for them.

    Returns
    -------
    np.ndarray
        Float32 array of shape (len(texts), 384).
    """
    encoder = get_encoder()
    # show_progress_bar=False keeps Streamlit output clean
    vectors = encoder.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize → cosine sim == dot product
    )
    return vectors.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Encode a single user query string into a 384-dim vector.

    Parameters
    ----------
    query : str
        The user's free-text question, e.g. "kenapa pelanggan komplain ukuran?"

    Returns
    -------
    np.ndarray
        Float32 array of shape (384,).
    """
    encoder = get_encoder()
    vector = encoder.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vector[0].astype(np.float32)
