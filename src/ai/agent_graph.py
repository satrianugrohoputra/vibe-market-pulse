"""
Agent Graph Module — LangGraph Agentic RAG
==========================================
Phase 2 implementation: a 2-node LangGraph workflow that
    1. Retrieves the top-K most relevant reviews from the vector store, and
    2. Synthesizes an executive report via a single Gemini call.

The Gemini call is delegated to a caller function (passed in) so we keep the
existing call_gemini_consultant logic untouched and reuse all its
domain/language-aware prompt engineering.

State flow:
    START → [retrieve_node] → [synthesize_node] → END

Phase 3 will extend this with conditional routing & self-critique loop.
"""

from __future__ import annotations

from typing import Callable, Optional, TypedDict


# ----------------------------------------------------------------------------
# Graph state schema
# ----------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    # === Inputs ===
    query: str                              # User-provided topic (may be empty)
    domain: str                             # Detected domain
    language: str                           # Detected language
    rule_corrected_count: int               # For Gemini context note
    top_k: int                              # Retrieval depth
    sentiment_filter: Optional[str]         # "Negative" by default

    # === Intermediate / outputs ===
    retrieved: list                         # List[dict] from vector_store.search
    report: str                             # Final markdown report
    error: Optional[str]                    # Set if a node fails


# ----------------------------------------------------------------------------
# Node factories
# ----------------------------------------------------------------------------
def _make_retrieve_node(vector_store):
    """Build the retrieval node bound to a specific vector store instance."""

    def retrieve_node(state: GraphState) -> GraphState:
        query = (state.get("query") or "").strip()
        # Default meta-query when user gives no topic
        if not query:
            query = "negative complaint problem issue bad disappointed kecewa jelek"

        try:
            results = vector_store.search(
                query=query,
                top_k=int(state.get("top_k", 10)),
                sentiment_filter=state.get("sentiment_filter"),
            )
            return {**state, "retrieved": results, "error": None}
        except Exception as exc:
            return {
                **state,
                "retrieved": [],
                "error": f"Retrieval failed: {exc}",
            }

    return retrieve_node


def _make_synthesize_node(
    gemini_caller: Callable[..., str],
    api_key: str,
    model_name: str,
):
    """Build the synthesis node that calls Gemini exactly once."""

    def synthesize_node(state: GraphState) -> GraphState:
        if state.get("error"):
            return {**state, "report": f"_⚠️ {state['error']}_"}

        retrieved = state.get("retrieved", []) or []
        review_texts = [
            r["text"] for r in retrieved
            if isinstance(r, dict) and str(r.get("text", "")).strip()
        ]

        if not review_texts:
            return {
                **state,
                "report": (
                    "_No relevant reviews retrieved for this query. "
                    "Try a different topic or upload more data._"
                ),
            }

        try:
            output = gemini_caller(
                api_key,
                review_texts,
                model_name=model_name,
                domain=state.get("domain", "general"),
                language=state.get("language", "English"),
                rule_corrected_count=int(state.get("rule_corrected_count", 0)),
            )
            return {**state, "report": output}
        except Exception as exc:
            return {
                **state,
                "report": f"_⚠️ Gemini synthesis failed: {exc}_",
            }

    return synthesize_node


# ----------------------------------------------------------------------------
# Graph builder
# ----------------------------------------------------------------------------
def build_agentic_rag_graph(
    vector_store,
    gemini_caller: Callable[..., str],
    api_key: str,
    model_name: str,
):
    """
    Compile a 2-node LangGraph:
        START → retrieve → synthesize → END
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", _make_retrieve_node(vector_store))
    graph.add_node(
        "synthesize",
        _make_synthesize_node(gemini_caller, api_key, model_name),
    )
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# ----------------------------------------------------------------------------
# Top-level convenience runner
# ----------------------------------------------------------------------------
def run_agentic_rag(
    vector_store,
    gemini_caller: Callable[..., str],
    api_key: str,
    model_name: str,
    query: str = "",
    domain: str = "general",
    language: str = "English",
    rule_corrected_count: int = 0,
    top_k: int = 10,
    sentiment_filter: Optional[str] = "Negative",
) -> dict:
    """
    Compile and execute the LangGraph workflow.

    Returns the final state dict containing:
        - retrieved: list[dict]    (top-K reviews used as context)
        - report: str              (Gemini-synthesized markdown)
        - error: Optional[str]
    """
    app = build_agentic_rag_graph(
        vector_store=vector_store,
        gemini_caller=gemini_caller,
        api_key=api_key,
        model_name=model_name,
    )

    initial_state: GraphState = {
        "query": query,
        "domain": domain,
        "language": language,
        "rule_corrected_count": rule_corrected_count,
        "top_k": top_k,
        "sentiment_filter": sentiment_filter,
        "retrieved": [],
        "report": "",
        "error": None,
    }

    return app.invoke(initial_state)
