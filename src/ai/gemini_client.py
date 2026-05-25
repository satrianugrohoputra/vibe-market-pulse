"""
src/ai/gemini_client.py
=======================
Thin Gemini wrapper untuk synthesis layer di Agentic RAG pipeline.

Design contract
---------------
* Gemini HANYA dipanggil di sini — bukan di embeddings, vector_store, atau clustering.
* Maksimal 2 panggilan per RAG run:
    1. synthesize_report()  — buat laporan aspek dari cluster evidence
    2. critique_report()    — self-check laporan vs source reviews (opsional/toggle)
* Model dipilih oleh user di app.py selectbox dan diteruskan sebagai `model_name`.
  Tidak ada hardcode di sini — model list tetap di app.py seperti sekarang.

Public API
----------
    synthesize_report(clusters, *, api_key, model_name, domain, language) -> str
    critique_report(report, evidence, *, api_key, model_name)             -> str
"""

from __future__ import annotations

from src.ai.clustering import ClusterResult

# ---------------------------------------------------------------------------
# Domain-specific aspect hints (mirror dari app.py agar prompt konsisten)
# ---------------------------------------------------------------------------
_DOMAIN_ASPECTS: dict[str, str] = {
    "clothing": (
        "  - Sizing & Fit\n"
        "  - Material & Fabric Quality\n"
        "  - Design & Style\n"
        "  - Color Accuracy\n"
        "  - Durability & Washing\n"
    ),
    "shoes": (
        "  - Comfort & Cushioning\n"
        "  - Sizing & Fit\n"
        "  - Sole & Grip Quality\n"
        "  - Durability & Wear\n"
        "  - Design & Aesthetics\n"
    ),
    "electronics": (
        "  - Battery & Power\n"
        "  - Screen & Display\n"
        "  - Sound & Audio Quality\n"
        "  - Build Quality & Durability\n"
        "  - Connectivity & Performance\n"
    ),
    "general": (
        "  - Product Quality\n"
        "  - Sizing & Fit\n"
        "  - Material Quality\n"
        "  - Design & Style\n"
        "  - Customer Service\n"
        "  - Shipping & Delivery\n"
        "  - Pricing & Value\n"
    ),
}


# ---------------------------------------------------------------------------
# Helper: format cluster evidence block for prompts
# ---------------------------------------------------------------------------

def _format_cluster_evidence(clusters: list[ClusterResult]) -> str:
    """
    Render clusters as a structured text block for injection into prompts.

    Example output:
        === CLUSTER 1 — sizing, fit, small, large, tight ===
        - "The shirt runs extremely small..."
        - "I ordered an L but it fits like a S..."
        ...

        === CLUSTER 2 — fabric, quality, thin, cheap ===
        ...
    """
    lines: list[str] = []
    for c in clusters:
        theme = ", ".join(c.top_terms) if c.top_terms else "general"
        lines.append(f"=== CLUSTER {c.cluster_id + 1} — {theme} ===")
        for review in c.reviews[:8]:           # cap at 8 per cluster → ~token budget
            clean = str(review).strip().replace("\n", " ")
            lines.append(f'- "{clean}"')
        lines.append("")                        # blank line between clusters
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 1: synthesize_report
# ---------------------------------------------------------------------------

def synthesize_report(
    clusters: list[ClusterResult],
    *,
    api_key: str,
    model_name: str,
    domain: str = "general",
    language: str = "English",
    user_query: str = "",
) -> str:
    """
    Gemini Call #1 — Generate an executive aspect-based BI report from clusters.

    Parameters
    ----------
    clusters : list[ClusterResult]
        Output of clustering.cluster_reviews().  Each cluster contains
        thematically similar negative reviews and their TF-IDF top terms.
    api_key : str
        Gemini API key from st.secrets.
    model_name : str
        Model name passed from app.py selectbox (e.g. "gemini-2.5-flash").
        NOT hardcoded here — caller decides.
    domain : str
        Detected product domain: 'clothing', 'shoes', 'electronics', 'general'.
    language : str
        Detected language: 'English' or 'Indonesian'.
    user_query : str
        Optional — the user's original question.  Included in the prompt so
        the report directly addresses the user's focus area.

    Returns
    -------
    str
        Markdown-formatted executive report.
    """
    from google import genai  # lazy import — only when Gemini is actually called

    client = genai.Client(api_key=api_key)

    evidence_block = _format_cluster_evidence(clusters)
    aspects_hint = _DOMAIN_ASPECTS.get(domain, _DOMAIN_ASPECTS["general"])

    lang_note = ""
    if language == "Indonesian":
        lang_note = (
            "\n**PENTING**: Ulasan di bawah dalam Bahasa Indonesia. "
            "Analisis kontennya tetapi tulis laporan dalam Bahasa Inggris. "
            "Terjemahkan kutipan kunci saat diperlukan.\n"
        )

    query_focus = ""
    if user_query.strip():
        query_focus = (
            f"\n**Fokus Pertanyaan User**: \"{user_query.strip()}\"\n"
            "Pastikan laporan menjawab pertanyaan ini secara langsung di Executive Summary.\n"
        )

    total_reviews = sum(c.size for c in clusters)
    n_clusters = len(clusters)

    prompt = f"""You are a senior Business Consultant specializing in e-commerce customer experience.

**Context**:
- Domain: {domain.title()}
- Language: {language}
- Total negative reviews analyzed: {total_reviews}
- Automatically identified complaint clusters: {n_clusters}
{lang_note}{query_focus}
The reviews below have been semantically retrieved (via vector similarity) and \
automatically clustered into {n_clusters} thematic groups using KMeans. \
Each cluster represents a distinct customer pain point pattern.

**Relevant aspect categories for {domain.title()} products**:
{aspects_hint}
Only include aspects with actual evidence — do not invent complaints.

---
CLUSTERED NEGATIVE REVIEWS:
{evidence_block}
---

Write a comprehensive Aspect-Based Sentiment Analysis report in clean Markdown \
using EXACTLY this structure:

## 📋 Executive Summary
2-3 sentences: overall sentiment snapshot + the most critical systemic issue \
identified across clusters. If the user asked a specific question, answer it here.

## 🔍 Categorized Pain Points by Cluster

For each cluster, use this sub-structure:
### 🔴 Cluster [N] — [Cluster Theme from top terms]
**Aspect**: [matched business aspect]
**Volume**: [N reviews in this cluster]
**Key Complaints**:
- [specific complaint with short quote]
- [specific complaint with short quote]
**Business Impact**: [1 sentence on why this matters]

## 🎯 High-Priority Action Items
Numbered list of 3-5 prioritized, actionable recommendations. \
Each must reference the cluster(s) it addresses and expected impact.

## 📊 Cluster Intelligence Summary
A brief table-style summary:
| Cluster | Theme | Volume | Priority |
|---------|-------|--------|----------|
[fill rows]
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return getattr(response, "text", str(response))


# ---------------------------------------------------------------------------
# Node 2: critique_report  (optional self-check node)
# ---------------------------------------------------------------------------

def critique_report(
    report: str,
    clusters: list[ClusterResult],
    *,
    api_key: str,
    model_name: str,
) -> str:
    """
    Gemini Call #2 — Self-critique: verify report claims against source evidence.

    This is the anti-hallucination node.  Gemini checks whether each major
    claim in the report is actually supported by the retrieved reviews.

    Parameters
    ----------
    report : str
        The Markdown report produced by synthesize_report().
    clusters : list[ClusterResult]
        The same clusters used to generate the report — used as ground truth.
    api_key : str
        Gemini API key.
    model_name : str
        Same model as synthesis (passed from app.py).

    Returns
    -------
    str
        A Markdown critique section that can be appended to the report UI.
        Format:
            ## ✅ Fact-Check Results
            - ✅ Claim: "..." → Supported by Cluster N
            - ⚠️ Claim: "..." → Partially supported / overstated
            - ❌ Claim: "..." → No direct evidence found

            ## 🔁 Refined Summary (if corrections needed)
            [corrected executive summary or "No corrections needed."]
    """
    from google import genai  # lazy import

    client = genai.Client(api_key=api_key)

    evidence_block = _format_cluster_evidence(clusters)

    prompt = f"""You are a rigorous fact-checker for an AI-generated business report.

Your task: verify that EVERY major claim in the report below is actually supported \
by the source review evidence provided. Flag any hallucinated, exaggerated, or \
unsupported claims.

---
ORIGINAL REPORT:
{report}

---
SOURCE EVIDENCE (retrieved reviews, grouped by cluster):
{evidence_block}
---

Produce your fact-check in this exact Markdown format:

## ✅ Fact-Check Results
For each significant claim in the report, write one line:
- ✅ **Supported**: "[claim summary]" → Evidence found in Cluster [N]
- ⚠️ **Overstated**: "[claim summary]" → Partially supported; actual evidence is weaker
- ❌ **Unsupported**: "[claim summary]" → No direct evidence in source reviews

## 🔁 Refined Summary
If any claims were overstated or unsupported, provide a corrected 2-3 sentence \
executive summary here.
If all claims are fully supported, write: "✅ All claims verified. No corrections needed."
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )
    return getattr(response, "text", str(response))
