"""
PostgreSQL + pgvector Store  (FUTURE / NOT YET WIRED IN)
========================================================

A *persistent* vector store backed by PostgreSQL with the `pgvector`
extension. This is the natural upgrade path from the current in-memory
NumPy store (`vector_store.py`) once the app needs data that survives
refreshes / restarts (e.g. on Streamlit Cloud, whose filesystem is
ephemeral).

⚠️  STATUS: This module is intentionally **dormant**. Nothing in `app.py`
    imports it, so it cannot affect the running app. All heavy dependencies
    are imported *lazily* (inside methods), so merely importing this file is
    safe even when psycopg / pgvector are not installed.

WHY pgvector (vs ChromaDB / NumPy):
    • NumPy (current): fast, zero-setup, but in-memory only — data is lost on
      every Streamlit rerun / container restart.
    • ChromaDB ephemeral: same volatility problem on Streamlit Cloud.
    • pgvector: embeddings live in your Postgres database alongside the rest
      of your data (users, uploads, reports). One database for everything,
      truly persistent, and production-grade.

────────────────────────────────────────────────────────────────────────
SETUP (when you're ready to use this — do NOT do this now):

1. Create a free Postgres with pgvector (e.g. Supabase or Neon).

2. Install the optional dependencies (NOT in requirements.txt yet, to keep
   the deployed app lean):

       pip install "psycopg[binary]>=3.1" pgvector

3. Put your connection string in `.streamlit/secrets.toml` (never commit it):

       # .streamlit/secrets.toml
       DATABASE_URL = "postgresql://user:password@host:5432/dbname"

4. Wire it into app.py only when you actually want persistence, e.g.:

       from src.ai.pgvector_store import PgVectorReviewStore
       store = PgVectorReviewStore(dsn=st.secrets["DATABASE_URL"])
       store.ensure_schema()
       store.index_dataframe(df, text_col="Review", dataset_id=df_hash, ...)
       results = store.search("shipping late", top_k=10,
                              sentiment_filter="Negative", dataset_id=df_hash)

The public API mirrors `ReviewVectorStore` (search / search_negatives) so it
can be swapped in with minimal changes.
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from typing import Any, Callable, Optional

# Embedding dimension for "paraphrase-multilingual-MiniLM-L12-v2".
EMBEDDING_DIM = 384
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBED_BATCH_SIZE = 64
TABLE_NAME = "review_embeddings"


class PgVectorReviewStore:
    """
    Persistent review vector store using PostgreSQL + pgvector.

    Designed as a drop-in conceptual replacement for the in-memory
    `ReviewVectorStore`, but with durable storage.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string, e.g.
        "postgresql://user:pass@host:5432/dbname".
    embedder:
        Optional pre-loaded sentence-transformers model. If omitted, the
        model is loaded lazily on first use (no Streamlit dependency, so this
        module stays framework-agnostic and testable).
    """

    def __init__(self, dsn: str, embedder: Optional[Any] = None):
        self.dsn = dsn
        self._embedder = embedder
        self.indexed_count = 0

    # ------------------------------------------------------------------
    # Lazy helpers
    # ------------------------------------------------------------------
    def _get_embedder(self):
        """Load the embedding model on first use (lazy)."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedder

    def _connect(self):
        """Open a psycopg connection with the pgvector adapter registered."""
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "psycopg and pgvector are required for the Postgres store. "
                'Install them with: pip install "psycopg[binary]>=3.1" pgvector'
            ) from exc

        conn = psycopg.connect(self.dsn)
        # Ensure the extension exists before registering the adapter
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)
        return conn

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def ensure_schema(self) -> None:
        """
        Create the extension, table, and an HNSW cosine index if they don't
        already exist. Safe to call repeatedly (idempotent).
        """
        create_table = f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id          BIGSERIAL PRIMARY KEY,
                dataset_id  TEXT NOT NULL,
                text        TEXT NOT NULL,
                sentiment   TEXT,
                rating      DOUBLE PRECISION,
                embedding   vector({EMBEDDING_DIM}) NOT NULL,
                created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """
        # HNSW index for fast approximate cosine search (pgvector >= 0.5).
        create_index = f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_embedding_hnsw
            ON {TABLE_NAME}
            USING hnsw (embedding vector_cosine_ops);
        """
        create_dataset_idx = f"""
            CREATE INDEX IF NOT EXISTS {TABLE_NAME}_dataset_idx
            ON {TABLE_NAME} (dataset_id);
        """
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(create_table)
                cur.execute(create_index)
                cur.execute(create_dataset_idx)
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def index_dataframe(
        self,
        df,
        text_col: str,
        dataset_id: str,
        sentiment_col: Optional[str] = None,
        rating_col: Optional[str] = None,
        replace_existing: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Embed and persist all valid rows of `df` for a given `dataset_id`.

        Parameters
        ----------
        dataset_id:
            A stable key (e.g. the dataframe hash from `compute_df_hash`) so
            multiple uploads can coexist in the same table without mixing.
        replace_existing:
            If True, delete any rows already stored under this dataset_id
            before inserting (a fresh rebuild per upload).

        Returns
        -------
        int : number of rows indexed.
        """
        import numpy as np  # local import keeps module import-safe

        embedder = self._get_embedder()

        # Filter out empty text rows
        valid_mask = df[text_col].fillna("").astype(str).str.strip().astype(bool)
        valid_df = df.loc[valid_mask].reset_index(drop=True)
        if len(valid_df) == 0:
            self.indexed_count = 0
            return 0

        texts = valid_df[text_col].astype(str).tolist()
        sentiments = (
            valid_df[sentiment_col].astype(str).tolist()
            if sentiment_col and sentiment_col in valid_df.columns
            else [None] * len(valid_df)
        )
        if rating_col and rating_col in valid_df.columns:
            import pandas as pd
            ratings = pd.to_numeric(valid_df[rating_col], errors="coerce").tolist()
        else:
            ratings = [None] * len(valid_df)

        conn = self._connect()
        try:
            if replace_existing:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {TABLE_NAME} WHERE dataset_id = %s;",
                        (dataset_id,),
                    )
                conn.commit()

            total = len(texts)
            inserted = 0
            for start in range(0, total, EMBED_BATCH_SIZE):
                end = min(start + EMBED_BATCH_SIZE, total)
                batch_texts = texts[start:end]

                embeddings = embedder.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)

                rows = [
                    (
                        dataset_id,
                        batch_texts[i],
                        sentiments[start + i],
                        (
                            None
                            if ratings[start + i] is None
                            or (isinstance(ratings[start + i], float)
                                and np.isnan(ratings[start + i]))
                            else float(ratings[start + i])
                        ),
                        embeddings[i],
                    )
                    for i in range(len(batch_texts))
                ]

                with conn.cursor() as cur:
                    cur.executemany(
                        f"""
                        INSERT INTO {TABLE_NAME}
                            (dataset_id, text, sentiment, rating, embedding)
                        VALUES (%s, %s, %s, %s, %s);
                        """,
                        rows,
                    )
                conn.commit()
                inserted += len(rows)

                if progress_callback is not None:
                    progress_callback(end, total)

            self.indexed_count = inserted
            return inserted
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        sentiment_filter: Optional[str] = None,
        dataset_id: Optional[str] = None,
    ) -> list[dict]:
        """
        Cosine semantic search via pgvector's `<=>` distance operator.

        Returns a list of dicts shaped exactly like `ReviewVectorStore.search`:
            {id, text, metadata: {sentiment, rating}, score}
        where score is cosine similarity in [0..1] (1 = perfect match).
        """
        if not query or not query.strip():
            return []

        import numpy as np

        embedder = self._get_embedder()
        query_vec = embedder.encode(
            [query], show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float32)[0]

        # Build a parameterized WHERE clause (prevents SQL injection)
        where_parts = []
        params: list = [query_vec]
        if sentiment_filter:
            where_parts.append("sentiment = %s")
            params.append(sentiment_filter)
        if dataset_id:
            where_parts.append("dataset_id = %s")
            params.append(dataset_id)
        where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
        params.append(int(top_k))

        sql = f"""
            SELECT id, text, sentiment, rating,
                   1 - (embedding <=> %s) AS score
            FROM {TABLE_NAME}
            {where_sql}
            ORDER BY embedding <=> %s
            LIMIT %s;
        """
        # The distance operator appears twice (SELECT score + ORDER BY), so we
        # need the query vector in both positions.
        ordered_params = [query_vec] + params[1:-1] + [query_vec, params[-1]]

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, ordered_params)
                rows = cur.fetchall()
        finally:
            conn.close()

        out: list[dict] = []
        for row in rows:
            _id, text, sentiment, rating, score = row
            out.append({
                "id": str(_id),
                "text": text,
                "metadata": {"sentiment": sentiment, "rating": rating},
                "score": max(0.0, min(1.0, float(score))),
            })
        return out

    def search_negatives(
        self,
        topic: str,
        top_k: int = 10,
        dataset_id: Optional[str] = None,
    ) -> list[dict]:
        """Convenience: targeted retrieval of Negative reviews for RAG."""
        if not topic or not topic.strip():
            topic = "negative complaint problem issue bad disappointed"
        return self.search(
            query=topic,
            top_k=top_k,
            sentiment_filter="Negative",
            dataset_id=dataset_id,
        )

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def delete_dataset(self, dataset_id: str) -> int:
        """Remove all rows for a given dataset_id. Returns rows deleted."""
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {TABLE_NAME} WHERE dataset_id = %s;",
                    (dataset_id,),
                )
                deleted = cur.rowcount
            conn.commit()
            return int(deleted)
        finally:
            conn.close()
