"""
Agent Graph Module — LangGraph Agentic RAG (Phase 3: Self-Critique Loop)
"""

from __future__ import annotations

import re
from typing import Callable, Literal, Optional, TypedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 2          # How many times validate→retrieve loops are allowed
DEFAULT_FALLBACK_QUERY = (
    "negative complaint problem issue bad disappointed kecewa jelek"
)

# Aspect keywords used by validate_report to check report completeness.
# Keys match what the Gemini prompt asks for; values are synonym lists.
ASPECT_CHECKS = {
    "executive_summary": ["executive summary", "## 📋"],
    "pain_points":       ["pain point", "## 🔍", "categorized"],
    "action_items":      ["action item", "## 🎯", "high-priority"],
}

# Minimum number of sections that must be present for validation to pass.
MIN_SECTIONS_REQUIRED = 3

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
                top_k=top_k,
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
                "step_log": log,
                "error": f"Retrieval failed: {exc}",
                "step_log": _log_step(
                    state, "3️⃣ Retrieve", f"❌ Failed: {exc}"
                ),
            }

    return retrieve_chunks


# ---------------------------------------------------------------------------
# Node D — cluster_aspects
# ---------------------------------------------------------------------------
def _make_cluster_aspects_node():
    """
    Groups retrieved reviews into thematic clusters using sentence embeddings
    + KMeans (no LLM required).

    If sklearn / numpy are unavailable or there are too few reviews, falls
    back to a single "all" cluster so the graph never breaks.

    Output: state["aspect_clusters"] = {cluster_label: [review_texts]}
    """

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

        # Build cluster preamble for Gemini context
        clusters = state.get("aspect_clusters") or {}
        cluster_preamble = ""
        if len(clusters) > 1:
            cluster_lines = []
            for label, items in clusters.items():
                cluster_lines.append(
                    f"  • {label.replace('_', ' ').title()} ({len(items)} reviews)"
                )
            cluster_preamble = (
                "\n\n**Pre-identified Thematic Clusters (from semantic analysis):**\n"
                + "\n".join(cluster_lines)
                + "\nPlease align your Categorized Pain Points with these clusters.\n"
            )

        retry = int(state.get("retry_count", 0))
        retry_note = ""
        if retry > 0:
            retry_note = (
                f"\n**Note**: This is synthesis attempt {retry + 1}. "
                "Please ensure ALL three required sections "
                "(Executive Summary, Categorized Pain Points, High-Priority Action Items) "
                "are fully included.\n"
            )

        try:
            # We inject cluster_preamble + retry_note into the domain field
            # by appending to it (Gemini caller will echo it in prompt header)
            enriched_domain = (
                state.get("effective_domain") or state.get("domain") or "general"
            )

            output = gemini_caller(
                api_key,
                balanced,
                model_name=model_name,
                domain=enriched_domain,
                language=state.get("language", "English"),
                rule_corrected_count=int(state.get("rule_corrected_count", 0)),
                # Pass cluster context via extra_context if supported,
                # otherwise it's silently ignored by the existing caller.
                extra_context=cluster_preamble + retry_note,
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

    return synthesize_report


# ---------------------------------------------------------------------------
# Node F — validate_report  (rule-based, no LLM)
# ---------------------------------------------------------------------------
def _make_validate_report_node():
    """
    Checks that the synthesized report meets minimum quality criteria:
    1. Contains all three required section headings.
    2. Has a minimum word count (not an empty / too-short response).
    3. Does not contain obvious failure markers (e.g. Gemini error strings).

    Sets state["validation_passed"] and state["validation_notes"].
    """
    MIN_WORD_COUNT = 40
    FAILURE_MARKERS = ["⚠️ Gemini synthesis failed", "No relevant reviews retrieved"]

    def validate_report(state: GraphState) -> GraphState:
        report = state.get("report") or ""
        log = list(state.get("step_log") or [])
        log.append("validate_report")

        issues: list[str] = []

        # Check for hard failure markers
        for marker in FAILURE_MARKERS:
            if marker in report:
                issues.append(f"Report contains error marker: '{marker}'")

        # Check required sections
        report_lower = report.lower()
        sections_found = 0
        for section_name, keywords in ASPECT_CHECKS.items():
            if any(kw.lower() in report_lower for kw in keywords):
                sections_found += 1
            else:
                issues.append(f"Missing section: {section_name}")

        if sections_found < MIN_SECTIONS_REQUIRED:
            issues.append(
                f"Only {sections_found}/{MIN_SECTIONS_REQUIRED} required sections found"
            )

        # Check minimum length
        word_count = len(report.split())
        if word_count < MIN_WORD_COUNT:
            issues.append(
                f"Report too short ({word_count} words, need ≥ {MIN_WORD_COUNT})"
            )

        passed = len(issues) == 0
        notes = "All checks passed." if passed else " | ".join(issues)

        return {
            **state,
            "validation_passed": passed,
            "validation_notes": notes,
            "step_log": log,
        }

    return validate_report


# ---------------------------------------------------------------------------
# Conditional router — after Node F
# ---------------------------------------------------------------------------
def _route_after_validate(state: GraphState) -> Literal["retrieve_chunks", "__end__"]:
    """
    Decides what happens after validation:
    - PASS or max retries exhausted → END
    - FAIL and retries remaining    → back to retrieve_chunks
    """
    if state.get("validation_passed"):
        return "__end__"

    retry = int(state.get("retry_count", 0))
    if retry >= MAX_RETRIES - 1:
        # Exhausted retries — accept the (imperfect) report
        return "__end__"

    return "retrieve_chunks"


# ---------------------------------------------------------------------------
# Retry counter — injected as a thin wrapper before retrieve_chunks on loop
# ---------------------------------------------------------------------------
def _increment_retry(state: GraphState) -> GraphState:
    """Bumps retry_count before re-entering retrieve_chunks."""
    return {**state, "retry_count": int(state.get("retry_count", 0)) + 1}


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
        "synthesize_report",
        _make_synthesize_report_node(gemini_caller, api_key, model_name),
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
    Compile and execute the Phase 3 LangGraph workflow.

    Returns the final state dict containing:
        - retrieved: list[dict]
        - clusters: list[ClusterInfo]
        - report: str
        - validation: dict
        - step_log: list of {step, info}
        - loop_count: int
        - error: Optional[str]
    """
    app = build_phase3_graph(
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
