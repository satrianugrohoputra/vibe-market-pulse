"""
Agent Graph Module — LangGraph Agentic RAG  (Phase 3)
======================================================
6-node workflow with self-critique loop:

    START
      │
      ▼
    [A] parse_query        — rule-based, no LLM
      │
      ▼
    [B] route_domain       — uses domain/language already in state
      │
      ▼
    [C] retrieve_chunks    — ChromaDB semantic search
      │
      ▼
    [D] cluster_aspects    — sklearn KMeans grouping, no LLM
      │
      ▼
    [E] synthesize_report  — single Gemini call
      │
      ▼
    [F] validate_report    — rule-based quality check
      │
      ├─ PASS ──────────────────────────────────► END
      │
      └─ FAIL (retry_count < MAX_RETRIES)
           │
           └─ refine_query ──────────────────────► [C] (with broader query)

Design rules:
- Gemini is called exactly ONCE per successful path (E only).
- If validation fails and we retry, we re-call Gemini once more (max 2 calls total).
- All non-LLM nodes are fast and free.
- `step_log` in state records node names for step-by-step UI rendering.
- `run_agentic_rag_p3()` is the new public entry point (keeps old `run_agentic_rag`
  as a thin alias for backward compatibility).
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


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------
class GraphState(TypedDict, total=False):
    # ── Inputs (set before first node) ──────────────────────────────────
    query: str                        # User-provided topic / focus area
    domain: str                       # Detected product domain
    language: str                     # "English" or "Indonesian"
    rule_corrected_count: int         # For Gemini context note
    top_k: int                        # Retrieval depth
    sentiment_filter: Optional[str]   # e.g. "Negative"

    # ── Intermediate ────────────────────────────────────────────────────
    parsed_query: str                 # Cleaned / expanded query (Node A)
    effective_domain: str             # Confirmed domain (Node B)
    retrieved: list                   # [{id, text, metadata, score}, …] (Node C)
    aspect_clusters: dict             # {label: [texts]} (Node D)
    retry_count: int                  # How many times we've looped back
    step_log: list                    # ["parse_query", "route_domain", …]

    # ── Outputs ─────────────────────────────────────────────────────────
    report: str                       # Final markdown report (Node E)
    validation_passed: bool           # True if Node F accepted the report
    validation_notes: str             # Why validation passed / failed
    error: Optional[str]              # Set if a node raises unexpectedly


# ---------------------------------------------------------------------------
# Node A — parse_query
# ---------------------------------------------------------------------------
def _make_parse_query_node():
    """
    Rule-based query normaliser.
    - Strips excess whitespace.
    - If query is empty → substitutes a broad negative-sentiment fallback.
    - Detects simple Indonesian queries and flags them (no LLM needed).
    """
    INDONESIAN_MARKERS = ["kecewa", "jelek", "kurang", "buruk", "lambat",
                          "rusak", "mengecewakan", "tidak", "ga", "gak"]

    def parse_query(state: GraphState) -> GraphState:
        raw = (state.get("query") or "").strip()

        if not raw:
            parsed = DEFAULT_FALLBACK_QUERY
        else:
            # Collapse whitespace, lowercase for processing
            parsed = " ".join(raw.split())

        # If query contains Indonesian words, expand it with English synonyms
        # so the multilingual embedder gets stronger signal
        lower = parsed.lower()
        if any(w in lower for w in INDONESIAN_MARKERS):
            parsed = parsed + " complaint problem issue bad negative"

        log = list(state.get("step_log") or [])
        log.append("parse_query")

        return {
            **state,
            "parsed_query": parsed,
            "step_log": log,
            "retry_count": state.get("retry_count", 0),
            "error": None,
        }

    return parse_query


# ---------------------------------------------------------------------------
# Node B — route_domain
# ---------------------------------------------------------------------------
def _make_route_domain_node():
    """
    Confirms the effective domain from state.
    In Phase 3 this is still rule-based (domain was already detected by
    detect_dataset_domain in app.py before the graph runs).
    Future versions can call an LLM here for multi-domain routing.
    """
    DOMAIN_ASPECT_MAP = {
        "clothing":    ["sizing", "fabric", "fit", "material", "washing"],
        "shoes":       ["sole", "comfort", "grip", "durability", "sizing"],
        "electronics": ["battery", "screen", "charging", "sound", "performance"],
        "general":     ["quality", "service", "shipping", "pricing", "design"],
    }

    def route_domain(state: GraphState) -> GraphState:
        domain = (state.get("domain") or "general").lower()
        if domain not in DOMAIN_ASPECT_MAP:
            domain = "general"

        log = list(state.get("step_log") or [])
        log.append("route_domain")

        return {
            **state,
            "effective_domain": domain,
            "step_log": log,
        }

    return route_domain


# ---------------------------------------------------------------------------
# Node C — retrieve_chunks
# ---------------------------------------------------------------------------
def _make_retrieve_chunks_node(vector_store):
    """
    Semantic retrieval from ChromaDB.
    On retry (retry_count > 0) the query is broadened automatically.
    """

    def retrieve_chunks(state: GraphState) -> GraphState:
        retry = int(state.get("retry_count", 0))
        query = state.get("parsed_query") or DEFAULT_FALLBACK_QUERY

        # On retry: widen the query to cast a broader net
        if retry > 0:
            query = query + " " + DEFAULT_FALLBACK_QUERY
            top_k = min(int(state.get("top_k", 10)) + 5 * retry, 25)
        else:
            top_k = int(state.get("top_k", 10))

        log = list(state.get("step_log") or [])
        log.append(f"retrieve_chunks (attempt {retry + 1})")

        try:
            results = vector_store.search(
                query=query,
                top_k=top_k,
                sentiment_filter=state.get("sentiment_filter"),
            )
            return {**state, "retrieved": results, "step_log": log, "error": None}
        except Exception as exc:
            return {
                **state,
                "retrieved": [],
                "step_log": log,
                "error": f"Retrieval failed: {exc}",
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

    def cluster_aspects(state: GraphState) -> GraphState:
        retrieved = state.get("retrieved") or []
        log = list(state.get("step_log") or [])
        log.append("cluster_aspects")

        texts = [
            r["text"] for r in retrieved
            if isinstance(r, dict) and str(r.get("text", "")).strip()
        ]

        if len(texts) < 4:
            # Too few texts to cluster meaningfully — one group is fine
            clusters = {"all_reviews": texts}
            return {**state, "aspect_clusters": clusters, "step_log": log}

        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import normalize

            # Reuse the vector store embedder to get embeddings
            # Import lazily so this node doesn't fail if embedder unavailable
            from src.ai.vector_store import get_embedding_model
            embedder = get_embedding_model()

            embeddings = embedder.encode(
                texts, show_progress_bar=False, convert_to_numpy=True
            )
            embeddings = normalize(embeddings)

            # Decide number of clusters: min(4, len//3) capped at 5
            n_clusters = max(2, min(5, len(texts) // 3))
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = km.fit_predict(embeddings)

            clusters: dict = {}
            for label, text in zip(labels, texts):
                key = f"cluster_{int(label) + 1}"
                clusters.setdefault(key, []).append(text)

        except Exception:
            # Fallback: single cluster
            clusters = {"all_reviews": texts}

        return {**state, "aspect_clusters": clusters, "step_log": log}

    return cluster_aspects


# ---------------------------------------------------------------------------
# Node E — synthesize_report  (the ONE Gemini call)
# ---------------------------------------------------------------------------
def _make_synthesize_report_node(
    gemini_caller: Callable[..., str],
    api_key: str,
    model_name: str,
):
    """
    Sends retrieved reviews to Gemini for synthesis.
    Passes cluster information via an enriched preamble so Gemini is aware
    of the thematic groupings identified in Node D.
    """

    def synthesize_report(state: GraphState) -> GraphState:
        if state.get("error"):
            log = list(state.get("step_log") or [])
            log.append("synthesize_report (skipped — error)")
            return {
                **state,
                "report": f"_⚠️ {state['error']}_",
                "step_log": log,
            }

        retrieved = state.get("retrieved") or []
        review_texts = [
            r["text"] for r in retrieved
            if isinstance(r, dict) and str(r.get("text", "")).strip()
        ]

        log = list(state.get("step_log") or [])
        log.append("synthesize_report")

        if not review_texts:
            return {
                **state,
                "report": (
                    "_No relevant reviews retrieved for this query. "
                    "Try a different topic or upload more data._"
                ),
                "step_log": log,
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
                review_texts,
                model_name=model_name,
                domain=enriched_domain,
                language=state.get("language", "English"),
                rule_corrected_count=int(state.get("rule_corrected_count", 0)),
                # Pass cluster context via extra_context if supported,
                # otherwise it's silently ignored by the existing caller.
                extra_context=cluster_preamble + retry_note,
            )
            return {**state, "report": output, "step_log": log}

        except TypeError:
            # Older call_gemini_consultant doesn't accept extra_context — fallback
            try:
                output = gemini_caller(
                    api_key,
                    review_texts,
                    model_name=model_name,
                    domain=state.get("effective_domain") or state.get("domain") or "general",
                    language=state.get("language", "English"),
                    rule_corrected_count=int(state.get("rule_corrected_count", 0)),
                )
                return {**state, "report": output, "step_log": log}
            except Exception as exc:
                return {
                    **state,
                    "report": f"_⚠️ Gemini synthesis failed: {exc}_",
                    "step_log": log,
                }
        except Exception as exc:
            return {
                **state,
                "report": f"_⚠️ Gemini synthesis failed: {exc}_",
                "step_log": log,
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


# ---------------------------------------------------------------------------
# Graph builder — Phase 3
# ---------------------------------------------------------------------------
def build_phase3_graph(
    vector_store,
    gemini_caller: Callable[..., str],
    api_key: str,
    model_name: str,
):
    """
    Compile the full 6-node Phase 3 LangGraph:

        START → parse_query → route_domain → retrieve_chunks
              → cluster_aspects → synthesize_report → validate_report
              → END   (or loop back to retrieve_chunks with bumped retry_count)
    """
    from langgraph.graph import StateGraph, END

    graph = StateGraph(GraphState)

    # ── Register nodes ──────────────────────────────────────────────────
    graph.add_node("parse_query",       _make_parse_query_node())
    graph.add_node("route_domain",      _make_route_domain_node())
    graph.add_node("retrieve_chunks",   _make_retrieve_chunks_node(vector_store))
    graph.add_node("cluster_aspects",   _make_cluster_aspects_node())
    graph.add_node(
        "synthesize_report",
        _make_synthesize_report_node(gemini_caller, api_key, model_name),
    )
    graph.add_node("validate_report",   _make_validate_report_node())
    # Thin node that bumps the counter before re-entering retrieve on loop
    graph.add_node("increment_retry",   _increment_retry)

    # ── Wire linear edges ───────────────────────────────────────────────
    graph.set_entry_point("parse_query")
    graph.add_edge("parse_query",      "route_domain")
    graph.add_edge("route_domain",     "retrieve_chunks")
    graph.add_edge("retrieve_chunks",  "cluster_aspects")
    graph.add_edge("cluster_aspects",  "synthesize_report")
    graph.add_edge("synthesize_report","validate_report")

    # ── Conditional edge from validate_report ───────────────────────────
    graph.add_conditional_edges(
        "validate_report",
        _route_after_validate,
        {
            "__end__":        END,
            "retrieve_chunks": "increment_retry",
        },
    )
    graph.add_edge("increment_retry", "retrieve_chunks")

    return graph.compile()


# ---------------------------------------------------------------------------
# Public runner — Phase 3
# ---------------------------------------------------------------------------
def run_agentic_rag_p3(
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
        retrieved          list[dict]   – reviews used as context
        aspect_clusters    dict         – {cluster_label: [texts]}
        report             str          – Gemini-synthesized markdown
        validation_passed  bool         – whether validation accepted the report
        validation_notes   str          – human-readable validation outcome
        step_log           list[str]    – ordered node names for UI rendering
        retry_count        int          – number of self-critique loops
        error              Optional[str]
    """
    app = build_phase3_graph(
        vector_store=vector_store,
        gemini_caller=gemini_caller,
        api_key=api_key,
        model_name=model_name,
    )

    initial_state: GraphState = {
        "query":                query,
        "domain":               domain,
        "language":             language,
        "rule_corrected_count": rule_corrected_count,
        "top_k":                top_k,
        "sentiment_filter":     sentiment_filter,
        # Intermediates
        "parsed_query":         "",
        "effective_domain":     domain,
        "retrieved":            [],
        "aspect_clusters":      {},
        "retry_count":          0,
        "step_log":             [],
        # Outputs
        "report":               "",
        "validation_passed":    False,
        "validation_notes":     "",
        "error":                None,
    }

    return app.invoke(initial_state)


# ---------------------------------------------------------------------------
# Backward-compatibility alias (Phase 2 callers still work)
# ---------------------------------------------------------------------------
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
    """Thin alias → delegates to run_agentic_rag_p3 (Phase 3 upgrade)."""
    return run_agentic_rag_p3(
        vector_store=vector_store,
        gemini_caller=gemini_caller,
        api_key=api_key,
        model_name=model_name,
        query=query,
        domain=domain,
        language=language,
        rule_corrected_count=rule_corrected_count,
        top_k=top_k,
        sentiment_filter=sentiment_filter,
    )
