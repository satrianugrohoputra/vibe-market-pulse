"""
src/ai/__init__.py
==================
AI subsystem public API — hybrid local vector processing + Gemini synthesis.

Stack
-----
    Local  (free): sentence-transformers MiniLM → ChromaDB → KMeans
    Cloud (hemat): Gemini synthesis + optional self-critique (1-2 calls/session)

Quickstart (di app.py)
----------------------
    from src.ai import run_agentic_consultant, AgentResult

    result: AgentResult = run_agentic_consultant(
        query="Kenapa pelanggan komplain ukuran?",
        all_negative_reviews=neg_reviews_list,
        api_key=api_key,
        model_name=selected_model,       # dari st.selectbox — tidak diubah
        domain=detected_domain,          # dari detect_dataset_domain()
        language=detected_language,
        top_k=20,
        enable_critique=False,           # True = tambah 1 Gemini call untuk fact-check
    )

    if result.success:
        for cluster in result.clusters:
            st.markdown(f"**{cluster.label}** — {cluster.size} reviews")
        st.markdown(result.report)
        if result.critique:
            st.markdown(result.critique)
    else:
        st.error(result.error)
"""

from src.ai.agent_engine import AgentResult, run_agentic_consultant
from src.ai.clustering import ClusterResult

__all__ = [
    "AgentResult",
    "ClusterResult",
    "run_agentic_consultant",
]
