"""
src/ai/agent_engine.py
======================
LangGraph Agentic RAG Orchestrator — inti dari AI Consultant feature.

Architecture
------------
Menggantikan logic random.sample() lama di app.py dengan pipeline semantik:

    START
      │
      ▼
    [Node 1: build_index_node]    ← embed semua negative reviews (lokal, gratis)
      │                             build ChromaDB in-memory index
      ▼
    [Node 2: retrieval_node]      ← embed user query → cosine search → top-20
      │                             (lokal, gratis)
      ▼
    [Node 3: clustering_node]     ← KMeans auto-k → 3-5 ClusterResult (lokal, gratis)
      │
      ▼
    [Node 4: synthesis_node]      ← Gemini Call #1: aspect-based BI report
      │
      ├─── enable_critique=True ──►[Node 5: critique_node]  ← Gemini Call #2
      │                                                         (opsional, toggle)
      ▼
    END  →  AgentResult

Cost model
----------
    Local layer  (nodes 1-3): $0, ~2-10 detik tergantung ukuran dataset
    Gemini layer (nodes 4-5): 1-2 API calls per session

Public API
----------
    run_agentic_consultant(
        query, all_negative_reviews, *,
        api_key, model_name, domain, language,
        top_k, enable_critique
    ) -> AgentResult

    AgentResult  (dataclass — bisa di-render langsung di Streamlit)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

import numpy as np
import pandas as pd

# LangGraph imports
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# Local modules
from src.ai.clustering import ClusterResult, cluster_reviews
from src.ai.embeddings import embed_query, embed_texts
from src.ai.gemini_client import critique_report, synthesize_report
from src.ai.vector_store import ReviewIndex, build_index


# ============================================================================
# AgentState — shared state yang mengalir antar node di graph
# ============================================================================

class AgentState(TypedDict):
    """
    Immutable-ish state dict yang dipass antar setiap node LangGraph.

    Fields diisi secara bertahap saat graph berjalan:
        - Input fields   : diisi sebelum graph.invoke()
        - Computed fields: diisi oleh masing-masing node
    """
    # --- Input fields (set oleh caller) ---
    query: str                              # user's free-text question
    all_negative_reviews: list[str]         # semua teks review negatif
    api_key: str                            # Gemini API key
    model_name: str                         # e.g. "gemini-2.5-flash"
    domain: str                             # "clothing" / "shoes" / "electronics" / "general"
    language: str                           # "English" / "Indonesian"
    top_k: int                              # jumlah review untuk di-retrieve (default: 20)
    enable_critique: bool                   # aktifkan Node 5 self-critique?

    # --- Computed fields (diisi saat graph berjalan) ---
    review_index: Optional[ReviewIndex]     # Node 1 output
    retrieved_reviews: list[str]            # Node 2 output
    retrieved_embeddings: Optional[np.ndarray]  # Node 2 output (untuk clustering)
    clusters: list[ClusterResult]           # Node 3 output
    report: str                             # Node 4 output
    critique: str                           # Node 5 output (kosong jika skip)
    error: str                              # error message jika ada yang gagal
    elapsed_seconds: float                  # total waktu eksekusi


# ============================================================================
# AgentResult — return type yang dikembalikan ke app.py
# ============================================================================

@dataclass
class AgentResult:
    """
    Output final dari run_agentic_consultant().

    Attributes
    ----------
    report : str
        Laporan Markdown dari Gemini synthesis node.
    critique : str
        Fact-check Markdown dari Gemini critique node.
        Kosong string "" jika enable_critique=False.
    clusters : list[ClusterResult]
        List cluster yang ditemukan — untuk ditampilkan di UI sebelum report.
    retrieved_count : int
        Jumlah review yang di-retrieve dari vector store.
    total_negative : int
        Total review negatif yang diindeks.
    elapsed_seconds : float
        Waktu total eksekusi pipeline (detik).
    error : str
        Pesan error jika pipeline gagal sebagian. Kosong jika sukses.
    success : bool
        True jika report berhasil di-generate (walau ada critique error).
    """
    report: str = ""
    critique: str = ""
    clusters: list[ClusterResult] = field(default_factory=list)
    retrieved_count: int = 0
    total_negative: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""
    success: bool = False


# ============================================================================
# Node implementations
# ============================================================================

def _node_build_index(state: AgentState) -> dict[str, Any]:
    """
    Node 1: Build Index
    -------------------
    Embed semua negative reviews dan simpan ke ChromaDB in-memory index.

    Catatan: embed_texts() berjalan lokal dengan MiniLM — tidak ada API call.
    Hasil di-cache di vector_store._INDEX_CACHE sehingga upload CSV yang sama
    tidak di-embed ulang.
    """
    reviews = state["all_negative_reviews"]

    if not reviews:
        return {
            "review_index": None,
            "error": "Tidak ada review negatif untuk diindeks.",
        }

    # Filter blank strings
    clean_reviews = [str(r).strip() for r in reviews if str(r).strip()]

    if not clean_reviews:
        return {
            "review_index": None,
            "error": "Semua review negatif kosong setelah cleaning.",
        }

    # Embed (lokal, di-cache oleh sentence-transformers)
    embeddings = embed_texts(clean_reviews)

    # Build ChromaDB index (di-cache by hash jika data sama)
    index = build_index(clean_reviews, embeddings)

    return {
        "review_index": index,
        "error": "",
    }


def _node_retrieval(state: AgentState) -> dict[str, Any]:
    """
    Node 2: Retrieval
    -----------------
    Embed user query → cosine similarity search → ambil top-k review
    paling relevan secara semantik.

    Jika query kosong (user tekan tombol tanpa isi pertanyaan), gunakan
    query default: "product quality issues complaints" agar tetap
    menghasilkan review yang representatif.
    """
    index: Optional[ReviewIndex] = state.get("review_index")
    if index is None or index.n_docs == 0:
        return {
            "retrieved_reviews": [],
            "retrieved_embeddings": None,
            "error": state.get("error", "Index belum dibangun."),
        }

    query = state["query"].strip()
    if not query:
        # Default fallback query — tetap semantik, tidak random
        query = "product quality issues and customer complaints"

    top_k = state.get("top_k", 20)

    # Embed query (lokal)
    query_vec = embed_query(query)

    # Retrieve top-k dari ChromaDB
    retrieved = index.query(query_vec, k=top_k)

    if not retrieved:
        return {
            "retrieved_reviews": [],
            "retrieved_embeddings": None,
            "error": "Tidak ada review yang ter-retrieve dari vector store.",
        }

    # Re-embed retrieved reviews untuk clustering
    # (embeddings mereka sudah ada di index tapi ChromaDB default tidak return-kan,
    #  jadi kita embed ulang — retrieved list kecil (≤20), cepat)
    retrieved_embeddings = embed_texts(retrieved)

    return {
        "retrieved_reviews": retrieved,
        "retrieved_embeddings": retrieved_embeddings,
        "error": "",
    }


def _node_clustering(state: AgentState) -> dict[str, Any]:
    """
    Node 3: Clustering
    ------------------
    KMeans auto-k (silhouette score) untuk mengelompokkan retrieved reviews
    menjadi 3-5 cluster tematik.

    Label cluster datang dari TF-IDF top-terms — tidak ada Gemini call di sini.
    """
    reviews = state.get("retrieved_reviews", [])
    embeddings = state.get("retrieved_embeddings")

    if not reviews or embeddings is None or len(embeddings) == 0:
        return {
            "clusters": [],
            "error": state.get("error", "Tidak ada review untuk di-cluster."),
        }

    clusters = cluster_reviews(
        texts=reviews,
        embeddings=embeddings,
        k=None,   # auto-select via silhouette score (range 2-5)
    )

    return {
        "clusters": clusters,
        "error": "",
    }


def _node_synthesis(state: AgentState) -> dict[str, Any]:
    """
    Node 4: Gemini Synthesis  (Gemini Call #1)
    -------------------------------------------
    Kirim cluster evidence ke Gemini untuk generate laporan BI aspek.

    Ini adalah satu-satunya mandatory Gemini call dalam pipeline.
    """
    clusters = state.get("clusters", [])

    if not clusters:
        return {
            "report": "❌ Tidak ada cluster yang tersedia untuk sintesis.",
            "error": "Clustering menghasilkan 0 cluster.",
        }

    try:
        report = synthesize_report(
            clusters=clusters,
            api_key=state["api_key"],
            model_name=state["model_name"],
            domain=state.get("domain", "general"),
            language=state.get("language", "English"),
            user_query=state.get("query", ""),
        )
        return {"report": report, "error": ""}

    except Exception as exc:
        return {
            "report": "",
            "error": f"Gemini synthesis gagal: {exc}",
        }


def _node_critique(state: AgentState) -> dict[str, Any]:
    """
    Node 5: Gemini Self-Critique  (Gemini Call #2 — opsional)
    ----------------------------------------------------------
    Fact-check laporan vs source evidence untuk eliminasi halusinasi.

    Node ini hanya dieksekusi jika enable_critique=True (toggle di UI).
    Jika synthesis gagal (report kosong), node ini di-skip otomatis.
    """
    report = state.get("report", "")
    clusters = state.get("clusters", [])

    if not report or not clusters:
        return {"critique": "", "error": state.get("error", "")}

    try:
        critique = critique_report(
            report=report,
            clusters=clusters,
            api_key=state["api_key"],
            model_name=state["model_name"],
        )
        return {"critique": critique, "error": ""}

    except Exception as exc:
        # Critique gagal = non-fatal, report tetap ditampilkan
        return {
            "critique": "",
            "error": f"Self-critique gagal (non-fatal): {exc}",
        }


# ============================================================================
# Conditional edge: apakah Node 5 dijalankan?
# ============================================================================

def _should_critique(state: AgentState) -> str:
    """
    Conditional edge function.

    Returns "critique" jika enable_critique=True DAN synthesis berhasil.
    Returns "end" jika skip.
    """
    if state.get("enable_critique", False) and state.get("report", ""):
        return "critique"
    return "end"


# ============================================================================
# Graph Builder
# ============================================================================

def _build_graph() -> Any:
    """
    Assemble dan compile LangGraph StateGraph.

    Topology:
        START → build_index → retrieval → clustering → synthesis
                                                            │
                                  ┌─────── enable_critique? ┤
                                  ▼ yes                    │ no
                               critique                    │
                                  │                        │
                                  └──────────► END ◄───────┘
    """
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("build_index", _node_build_index)
    graph.add_node("retrieval",   _node_retrieval)
    graph.add_node("clustering",  _node_clustering)
    graph.add_node("synthesis",   _node_synthesis)
    graph.add_node("critique",    _node_critique)

    # Linear edges
    graph.add_edge(START,         "build_index")
    graph.add_edge("build_index", "retrieval")
    graph.add_edge("retrieval",   "clustering")
    graph.add_edge("clustering",  "synthesis")

    # Conditional edge: synthesis → critique OR end
    graph.add_conditional_edges(
        "synthesis",
        _should_critique,
        {
            "critique": "critique",
            "end":      END,
        },
    )

    # Critique always leads to END
    graph.add_edge("critique", END)

    return graph.compile()


# Module-level compiled graph (built once, reused)
_COMPILED_GRAPH = _build_graph()


# ============================================================================
# Public API — dipanggil dari app.py
# ============================================================================

def run_agentic_consultant(
    query: str,
    all_negative_reviews: list[str],
    *,
    api_key: str,
    model_name: str,
    domain: str = "general",
    language: str = "English",
    top_k: int = 20,
    enable_critique: bool = False,
) -> AgentResult:
    """
    Entry point untuk Agentic RAG pipeline.

    Menggantikan collect_negative_samples() + call_gemini_consultant() di app.py.
    Dipanggil saat user tekan tombol atau submit chat input di AI Consultant section.

    Parameters
    ----------
    query : str
        User's free-text question, e.g. "Kenapa pelanggan komplain ukuran Adidas?"
        Boleh kosong — pipeline akan gunakan default semantic query.
    all_negative_reviews : list[str]
        SEMUA review negatif dari uploaded data (bukan sample 10).
        Diperoleh dari: predicted_df[predicted_df["Predicted_IND"] == 0][text_col].tolist()
    api_key : str
        Gemini API key dari st.secrets["GEMINI_API_KEY"].
    model_name : str
        Dari selectbox di app.py, e.g. "gemini-2.5-flash".
        TIDAK diubah di sini — respek constraint "jangan ubah model list".
    domain : str
        Output detect_dataset_domain() dari app.py.
    language : str
        Output detect_dataset_domain() dari app.py.
    top_k : int
        Jumlah review yang diambil dari vector store. Default 20.
        Lebih besar = lebih representatif tapi sedikit lebih lambat di clustering.
    enable_critique : bool
        Jika True, jalankan Gemini self-critique node (Gemini Call #2).
        Default False untuk hemat quota free tier.

    Returns
    -------
    AgentResult
        Berisi: report, critique, clusters, metadata, error flag.

    Example usage di app.py
    -----------------------
        from src.ai import run_agentic_consultant, AgentResult

        neg_reviews = predicted_df[
            predicted_df["Predicted_IND"] == 0
        ][user_text_col].dropna().astype(str).tolist()

        result: AgentResult = run_agentic_consultant(
            query=user_question,
            all_negative_reviews=neg_reviews,
            api_key=api_key,
            model_name=selected_model,    # dari st.selectbox
            domain=detected_domain,
            language=detected_language,
            top_k=20,
            enable_critique=enable_critique_toggle,
        )

        if result.success:
            # Tampilkan clusters
            for cluster in result.clusters:
                st.markdown(f"**{cluster.label}** ({cluster.size} reviews)")
            # Tampilkan report
            st.markdown(result.report)
            # Tampilkan critique jika ada
            if result.critique:
                st.markdown(result.critique)
        else:
            st.error(result.error)
    """
    t_start = time.perf_counter()

    initial_state: AgentState = {
        # Input
        "query":                 query,
        "all_negative_reviews":  all_negative_reviews,
        "api_key":               api_key,
        "model_name":            model_name,
        "domain":                domain,
        "language":              language,
        "top_k":                 top_k,
        "enable_critique":       enable_critique,
        # Computed (initialized to empty/None — nodes will fill these)
        "review_index":          None,
        "retrieved_reviews":     [],
        "retrieved_embeddings":  None,
        "clusters":              [],
        "report":                "",
        "critique":              "",
        "error":                 "",
        "elapsed_seconds":       0.0,
    }

    try:
        final_state: AgentState = _COMPILED_GRAPH.invoke(initial_state)
    except Exception as exc:
        return AgentResult(
            error=f"LangGraph pipeline error: {exc}",
            success=False,
            elapsed_seconds=time.perf_counter() - t_start,
            total_negative=len(all_negative_reviews),
        )

    elapsed = time.perf_counter() - t_start

    # Determine success: report harus ada dan error utama tidak terjadi
    report = final_state.get("report", "")
    error  = final_state.get("error", "")
    success = bool(report) and "gagal" not in error.lower()

    return AgentResult(
        report=report,
        critique=final_state.get("critique", ""),
        clusters=final_state.get("clusters", []),
        retrieved_count=len(final_state.get("retrieved_reviews", [])),
        total_negative=len(all_negative_reviews),
        elapsed_seconds=round(elapsed, 2),
        error=error,
        success=success,
    )
