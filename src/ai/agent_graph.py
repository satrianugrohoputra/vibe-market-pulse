"""
Agent Graph Module — LangGraph Agentic RAG (Phase 3: Self-Critique Loop)
========================================================================
A 6-node LangGraph workflow with rule-based validation and conditional
loop-back for self-critique. Gemini is still called only ONCE per successful
run (never per-node), so it stays free-tier-friendly.

Workflow:
    START → parse_query → route_domain → retrieve → cluster_aspects
                                              ▲           │
                                              │           ▼
                                   refine_query   synthesize → validate ──→ END
                                              ▲                       │
                                              │  (loop back if missed │
                                              │   clusters & loops <2)│
                                              └───────────────────────┘

Nodes (rule-based unless marked LLM):
    1. parse_query       — normalize/clean the user topic, decide if it's
                           generic or specific
    2. route_domain      — pass-through the domain/language detected upstream
                           (already done in app.py before calling the graph)
    3. retrieve          — top-K cosine search in ChromaDB
    4. cluster_aspects   — TF-IDF + KMeans on retrieved texts → N aspect groups
    5. synthesize        — [LLM] Gemini call once, with balanced cluster sample
    6. validate_report   — keyword-coverage check vs. cluster keywords
    (conditional)
    7. refine_query      — broaden/swap query terms if validation failed,
                           then loop back to retrieve (max 2 loops)

State carries `step_log` so the UI can render a pretty step-by-step trace.

Public API (back-compat with Phase 2 caller in app.py):
    run_agentic_rag(...) -> dict with keys: retrieved, report, error,
                                            clusters, step_log, validation,
                                            loop_count
"""

from __future__ import annotations

from typing import Callable, Optional, TypedDict

from .aspect_clusters import (
    cluster_reviews,
    balanced_sample_from_clusters,
    validate_report_coverage,
    ClusterInfo,
)

# Top-level imports — fail-fast if LangGraph isn't installed.
try:
    from langgraph.graph import StateGraph, END
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LangGraph is required for the Agentic RAG workflow. "
        "Install it with: pip install langgraph>=0.2.0"
    ) from exc


# ============================================================================
# Constants — workflow tuning knobs
# ============================================================================
MAX_LOOPS = 2                # at most 2 retries before forcing END
DEFAULT_N_CLUSTERS = 4       # KMeans target on retrieved reviews
GENERIC_FALLBACK_QUERY = (
    "negative complaint problem issue bad disappointed kecewa jelek lambat rusak"
)


# ============================================================================
# Graph state schema
# ============================================================================
class GraphState(TypedDict, total=False):
    # ── Inputs ──
    query: str                              # User-provided topic (may be empty)
    original_query: str                     # Preserved for trace; never mutated
    domain: str                             # Detected domain
    language: str                           # Detected language
    rule_corrected_count: int               # For Gemini context note
    top_k: int                              # Retrieval depth
    sentiment_filter: Optional[str]         # "Negative" by default

    # ── Intermediate ──
    retrieved: list                         # List[dict] from vector_store.search
    clusters: list                          # List[ClusterInfo] from KMeans
    balanced_samples: list                  # Texts after round-robin pick
    is_query_generic: bool                  # parse_query → True if topic empty

    # ── Outputs ──
    report: str                             # Final markdown report
    validation: dict                        # output of validate_report_coverage
    step_log: list                          # ordered list of step records (UI)
    loop_count: int                         # how many self-critique loops ran
    error: Optional[str]                    # set if a node fails


# ============================================================================
# Helpers
# ============================================================================
def _log_step(state: GraphState, name: str, info: str) -> list:
    """Return a NEW step_log list with a step appended (immutable update)."""
    log = list(state.get("step_log") or [])
    log.append({"step": name, "info": info})
    return log


# ============================================================================
# Node 1 — parse_query (rule-based, no LLM)
# ============================================================================
def parse_query_node(state: GraphState) -> GraphState:
    raw = (state.get("query") or "").strip()
    is_generic = len(raw) == 0

    # Normalize: collapse whitespace, lowercase doesn't matter (embedder
    # handles casing) but trimming + length check is enough.
    normalized = raw if not is_generic else GENERIC_FALLBACK_QUERY

    return {
        **state,
        "query": normalized,
        "original_query": raw,
        "is_query_generic": is_generic,
        "step_log": _log_step(
            state,
            "1️⃣ Parse Query",
            f"{'Empty topic — using generic fallback' if is_generic else f'Topic: {raw!r}'}",
        ),
    }


# ============================================================================
# Node 2 — route_domain (pass-through, already detected in app.py)
# ============================================================================
def route_domain_node(state: GraphState) -> GraphState:
    domain = state.get("domain", "general")
    language = state.get("language", "English")
    return {
        **state,
        "step_log": _log_step(
            state,
            "2️⃣ Route Domain",
            f"Domain: {domain.title()} · Language: {language}",
        ),
    }


# ============================================================================
# Node 3 — retrieve (ChromaDB cosine search)
# ============================================================================
def _make_retrieve_node(vector_store):
    def retrieve_node(state: GraphState) -> GraphState:
        query = (state.get("query") or "").strip() or GENERIC_FALLBACK_QUERY
        try:
            results = vector_store.search(
                query=query,
                top_k=int(state.get("top_k", 10)),
                sentiment_filter=state.get("sentiment_filter"),
            )
            return {
                **state,
                "retrieved": results,
                "error": None,
                "step_log": _log_step(
                    state,
                    "3️⃣ Retrieve",
                    f"Top-{len(results)} reviews fetched (cosine similarity)",
                ),
            }
        except Exception as exc:
            return {
                **state,
                "retrieved": [],
                "error": f"Retrieval failed: {exc}",
                "step_log": _log_step(
                    state, "3️⃣ Retrieve", f"❌ Failed: {exc}"
                ),
            }

    return retrieve_node


# ============================================================================
# Node 4 — cluster_aspects (sklearn KMeans, no LLM)
# ============================================================================
def cluster_aspects_node(state: GraphState) -> GraphState:
    retrieved = state.get("retrieved", []) or []
    texts = [
        str(r.get("text", "")).strip()
        for r in retrieved
        if isinstance(r, dict) and str(r.get("text", "")).strip()
    ]

    if len(texts) < 6:
        # Too few reviews to cluster meaningfully; pass through unchanged
        return {
            **state,
            "clusters": [],
            "balanced_samples": texts,
            "step_log": _log_step(
                state,
                "4️⃣ Cluster Aspects",
                f"Only {len(texts)} reviews — clustering skipped, "
                "using all reviews as-is",
            ),
        }

    clusters = cluster_reviews(texts, n_clusters=DEFAULT_N_CLUSTERS)

    if not clusters:
        return {
            **state,
            "clusters": [],
            "balanced_samples": texts,
            "step_log": _log_step(
                state,
                "4️⃣ Cluster Aspects",
                "Clustering failed — using all retrieved reviews",
            ),
        }

    # Round-robin pick to give Gemini balanced exposure to every aspect
    balanced = balanced_sample_from_clusters(
        clusters,
        texts,
        target_size=len(texts),  # use all available; round-robin just orders
    )

    cluster_summary = " · ".join(
        f"#{c['cluster_id']}({c['size']}): {', '.join(c['keywords'][:3])}"
        for c in clusters
    )
    return {
        **state,
        "clusters": clusters,
        "balanced_samples": balanced,
        "step_log": _log_step(
            state,
            "4️⃣ Cluster Aspects",
            f"Found {len(clusters)} aspect groups → {cluster_summary}",
        ),
    }


# ============================================================================
# Node 5 — synthesize (Gemini, ONLY LLM call)
# ============================================================================
def _make_synthesize_node(
    gemini_caller: Callable[..., str],
    api_key: str,
    model_name: str,
):
    def synthesize_node(state: GraphState) -> GraphState:
        if state.get("error"):
            return {
                **state,
                "report": f"_⚠️ {state['error']}_",
                "step_log": _log_step(
                    state, "5️⃣ Synthesize", "Skipped due to upstream error"
                ),
            }

        # Prefer balanced cluster samples; fall back to raw retrieved texts
        balanced = state.get("balanced_samples") or []
        if not balanced:
            balanced = [
                str(r.get("text", "")).strip()
                for r in (state.get("retrieved") or [])
                if isinstance(r, dict) and str(r.get("text", "")).strip()
            ]

        if not balanced:
            return {
                **state,
                "report": (
                    "_No relevant reviews retrieved for this query. "
                    "Try a different topic or upload more data._"
                ),
                "step_log": _log_step(
                    state, "5️⃣ Synthesize", "No reviews to synthesize"
                ),
            }

        try:
            output = gemini_caller(
                api_key,
                balanced,
                model_name=model_name,
                domain=state.get("domain", "general"),
                language=state.get("language", "English"),
                rule_corrected_count=int(state.get("rule_corrected_count", 0)),
            )
            loop = int(state.get("loop_count", 0))
            label = "5️⃣ Synthesize" if loop == 0 else f"5️⃣ Synthesize (retry #{loop})"
            return {
                **state,
                "report": output,
                "step_log": _log_step(
                    state, label, f"Gemini called once with {len(balanced)} reviews"
                ),
            }
        except Exception as exc:
            return {
                **state,
                "report": f"_⚠️ Gemini synthesis failed: {exc}_",
                "step_log": _log_step(
                    state, "5️⃣ Synthesize", f"❌ Failed: {exc}"
                ),
            }

    return synthesize_node


# ============================================================================
# Node 6 — validate_report (rule-based, no LLM)
# ============================================================================
def validate_report_node(state: GraphState) -> GraphState:
    clusters: list[ClusterInfo] = state.get("clusters") or []
    report = state.get("report") or ""

    # If Gemini failed earlier, don't bother validating
    if state.get("error") or report.startswith("_⚠️"):
        return {
            **state,
            "validation": {
                "is_valid": True,  # treat as terminal — no point looping
                "coverage_pct": 0.0,
                "covered_clusters": [],
                "missed_clusters": [],
            },
            "step_log": _log_step(
                state, "6️⃣ Validate", "Skipped (upstream error)"
            ),
        }

    result = validate_report_coverage(report, clusters)
    miss_str = (
        ", ".join(
            f"#{c['cluster_id']}({c['keywords'][0]})"
            for c in result["missed_clusters"]
        ) or "none"
    )
    info = (
        f"Coverage: {result['coverage_pct'] * 100:.0f}% "
        f"({len(result['covered_clusters'])}/{len(clusters)} clusters covered) · "
        f"Missed: {miss_str}"
    )
    return {
        **state,
        "validation": result,
        "step_log": _log_step(state, "6️⃣ Validate", info),
    }


# ============================================================================
# Node 7 — refine_query (rule-based, no LLM)
# ============================================================================
def refine_query_node(state: GraphState) -> GraphState:
    """
    Build a NEW query that mentions the missed-cluster keywords explicitly,
    so the next retrieve call pulls reviews that actually mention those topics.
    """
    validation = state.get("validation") or {}
    missed: list[ClusterInfo] = validation.get("missed_clusters") or []

    # Collect top-2 keywords from each missed cluster
    refine_terms: list[str] = []
    for cluster in missed:
        refine_terms.extend(cluster.get("keywords", [])[:2])

    original = state.get("original_query") or ""

    if refine_terms:
        new_query = " ".join(
            [original] + refine_terms + ["complaint problem issue"]
        ).strip()
    else:
        # No specific misses — just broaden with the generic fallback
        new_query = f"{original} {GENERIC_FALLBACK_QUERY}".strip()

    new_loop_count = int(state.get("loop_count", 0)) + 1
    return {
        **state,
        "query": new_query,
        "loop_count": new_loop_count,
        "error": None,  # clear so next retrieve is fresh
        "step_log": _log_step(
            state,
            f"🔁 Refine Query (loop {new_loop_count}/{MAX_LOOPS})",
            f"New query: {new_query!r}",
        ),
    }


# ============================================================================
# Conditional edge — decide next step after validation
# ============================================================================
def route_after_validation(state: GraphState) -> str:
    """
    Conditional router that runs after validate_report_node.

    Returns:
        "refine"  → loop back to refine_query → retrieve again
        "end"     → finish the workflow
    """
    validation = state.get("validation") or {}
    is_valid = validation.get("is_valid", True)
    loop_count = int(state.get("loop_count", 0))

    if not is_valid and loop_count < MAX_LOOPS:
        return "refine"
    return "end"


# ============================================================================
# Graph builder
# ============================================================================
def build_agentic_rag_graph(
    vector_store,
    gemini_caller: Callable[..., str],
    api_key: str,
    model_name: str,
):
    """
    Compile the 6-node LangGraph with conditional self-critique loop.

    START → parse_query → route_domain → retrieve → cluster_aspects
                                              ▲           │
                                              │           ▼
                                       refine_query   synthesize → validate ──→ END
                                              ▲                          │
                                              └──────────(if invalid)────┘
    """
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("parse_query", parse_query_node)
    graph.add_node("route_domain", route_domain_node)
    graph.add_node("retrieve", _make_retrieve_node(vector_store))
    graph.add_node("cluster_aspects", cluster_aspects_node)
    graph.add_node(
        "synthesize",
        _make_synthesize_node(gemini_caller, api_key, model_name),
    )
    graph.add_node("validate", validate_report_node)
    graph.add_node("refine_query", refine_query_node)

    # Linear backbone
    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query", "route_domain")
    graph.add_edge("route_domain", "retrieve")
    graph.add_edge("retrieve", "cluster_aspects")
    graph.add_edge("cluster_aspects", "synthesize")
    graph.add_edge("synthesize", "validate")

    # Conditional self-critique loop
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "refine": "refine_query",
            "end": END,
        },
    )

    # After refining, loop back to retrieve (which will re-cluster, re-synthesize)
    graph.add_edge("refine_query", "retrieve")

    return graph.compile()


# ============================================================================
# Top-level convenience runner (back-compat with app.py)
# ============================================================================
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
        - retrieved: list[dict]
        - clusters: list[ClusterInfo]
        - report: str
        - validation: dict
        - step_log: list of {step, info}
        - loop_count: int
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
        "original_query": "",
        "domain": domain,
        "language": language,
        "rule_corrected_count": rule_corrected_count,
        "top_k": top_k,
        "sentiment_filter": sentiment_filter,
        "retrieved": [],
        "clusters": [],
        "balanced_samples": [],
        "is_query_generic": False,
        "report": "",
        "validation": {},
        "step_log": [],
        "loop_count": 0,
        "error": None,
    }

    # `recursion_limit` must be high enough to allow MAX_LOOPS retries through
    # 4 nodes (refine→retrieve→cluster→synthesize→validate).
    return app.invoke(initial_state, config={"recursion_limit": 50})
