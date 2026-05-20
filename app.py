"""
Hybrid E-Commerce Sentiment Analyzer
====================================
A Streamlit app that combines a Traditional NLP ML pipeline (TF-IDF + Logistic
Regression) with Generative AI (Gemini 2.5 Flash) for executive-level insights
on e-commerce review datasets.

Base dataset columns (Women's E-Commerce Clothing Reviews):
    - "Review Text"     : free-text customer review (HARDCODED for base training)
    - "Recommended IND" : 1 = Positive (recommended), 0 = Negative (not recommended)

User-uploaded CSVs use dynamic, case-insensitive text column detection.
"""

from __future__ import annotations

import io
import random
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
BASE_DATASET_PATH = "ecommercereviews.csv"

# HARDCODED columns for the base training pipeline. Do NOT change these.
TEXT_COL = "Review Text"
TARGET_COL = "Recommended IND"

# Dynamic text column candidates for USER UPLOADS only (case-insensitive match).
USER_TEXT_COLUMN_CANDIDATES = [
    "review text",
    "review",
    "text",
    "comment",
    "content",
    "description",
]

NEGATIVE_LABEL = 0  # not recommended
POSITIVE_LABEL = 1  # recommended

LABEL_MAP = {0: "Negative", 1: "Positive"}

# ----------------------------------------------------------------------------
# Auto-Routing: Domain & Language Keyword Dictionaries
# ----------------------------------------------------------------------------
# These keyword sets power the meta-classifier (detect_dataset_domain).
# Add/extend them as new domains or languages are supported.
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "clothing":    ["fabric", "dress", "size", "wear", "fit", "shirt"],
    "shoes":       ["sole", "running", "shoe", "sneaker", "comfortable", "tight", "grippy"],
    "electronics": ["battery", "screen", "charge", "sound", "button", "device"],
}

# Indonesian indicator keywords for language detection.
INDONESIAN_KEYWORDS: list[str] = [
    "bagus", "jelek", "kecewa", "kurang", "mantap", "baju", "sepatu",
]


# ----------------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid E-Commerce Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Cached: Train Base ML Pipeline (uses HARDCODED columns)
# ============================================================================
@st.cache_resource(show_spinner="Training base sentiment model...")
def train_base_pipeline() -> dict:
    """
    Loads the base CSV using the HARDCODED columns 'Review Text' and
    'Recommended IND'. Cleans NaN values, trains a TF-IDF + LogReg pipeline
    on an 80/20 split, and returns the trained pipeline plus eval info.
    """
    df = pd.read_csv(BASE_DATASET_PATH)

    if TEXT_COL not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COL}' not found in base dataset. "
            f"Available: {list(df.columns)}"
        )
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Column '{TARGET_COL}' not found in base dataset. "
            f"Available: {list(df.columns)}"
        )

    # CRITICAL: NaN handling
    df = df.dropna(subset=[TARGET_COL])
    df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Keep only valid binary labels
    df = df[df[TARGET_COL].isin([0, 1])].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No usable rows after cleaning the base dataset.")

    X = df[TEXT_COL].tolist()
    y = df[TARGET_COL].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words="english",
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        zero_division=0,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        zero_division=0,
        output_dict=True,
    )

    base_negatives = [
        X_test[i] for i in range(len(X_test))
        if y_pred[i] == NEGATIVE_LABEL and str(X_test[i]).strip()
    ]

    return {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "report": report_text,
        "report_dict": report_dict,
        "classes": ["Negative", "Positive"],
        "base_negative_samples": base_negatives,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "domain": "clothing",
    }


# ============================================================================
# Dynamic Domain Pipeline — Trains on-the-fly from uploaded data
# ============================================================================
def train_domain_pipeline(
    df: pd.DataFrame,
    text_col: str,
    domain: str,
) -> Optional[dict]:
    """
    Train a TF-IDF + Logistic Regression pipeline dynamically from an uploaded
    dataset. Uses the rating column to generate binary labels:
        rating >= 4 → Positive (1)
        rating <= 2 → Negative (0)
        rating == 3 → excluded (ambiguous)

    Returns a dict with the same structure as train_base_pipeline(), or None
    if the dataset lacks a usable rating column or has insufficient labeled data.
    """
    # --- Find rating column ---
    rating_col: Optional[str] = None
    for col in df.columns:
        name = str(col).strip().lower()
        if "rating" in name or "score" in name or "star" in name:
            rating_col = col
            break

    if rating_col is None:
        return None  # Cannot train without rating signal

    # --- Build labeled subset ---
    work = df[[text_col, rating_col]].copy()
    work[text_col] = work[text_col].fillna("").astype(str)
    work[rating_col] = pd.to_numeric(work[rating_col], errors="coerce")
    work = work.dropna(subset=[rating_col])

    # Derive binary labels from rating
    work["_label"] = -1  # placeholder
    work.loc[work[rating_col] >= 4, "_label"] = POSITIVE_LABEL
    work.loc[work[rating_col] <= 2, "_label"] = NEGATIVE_LABEL

    # Drop ambiguous (rating == 3) and empty text
    work = work[work["_label"].isin([0, 1])].reset_index(drop=True)
    work = work[work[text_col].str.strip().astype(bool)].reset_index(drop=True)

    if len(work) < 20:
        return None  # Not enough data to train a meaningful model

    X = work[text_col].tolist()
    y = work["_label"].astype(int).tolist()

    # Check class balance — need at least 2 of each class
    from collections import Counter
    counts = Counter(y)
    if counts.get(0, 0) < 2 or counts.get(1, 0) < 2:
        return None

    # Stratified split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y,
        )
    except ValueError:
        return None  # Stratification impossible

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        zero_division=0,
    )
    report_dict = classification_report(
        y_test, y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        zero_division=0,
        output_dict=True,
    )

    neg_samples = [
        X_test[i] for i in range(len(X_test))
        if y_pred[i] == NEGATIVE_LABEL and str(X_test[i]).strip()
    ]

    return {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "report": report_text,
        "report_dict": report_dict,
        "classes": ["Negative", "Positive"],
        "base_negative_samples": neg_samples,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "domain": domain,
    }


# ============================================================================
# Dynamic Text Column Detection (User Uploads ONLY)
# ============================================================================
def find_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Locate a usable text column in a USER-UPLOADED CSV.

    Strategy: case-insensitive match against USER_TEXT_COLUMN_CANDIDATES.
    Returns the FIRST matched column (preserving its original casing in the
    DataFrame). Returns None if no candidate is found.
    """
    # Build a map: lowercase column name -> original column name
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}

    for candidate in USER_TEXT_COLUMN_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    return None


# ============================================================================
# Auto-Routing Ensemble Architecture — Domain & Language Meta-Classifier
# ============================================================================
def detect_dataset_domain(df: pd.DataFrame, text_col: str) -> Tuple[str, str]:
    """
    Lightweight rule-based meta-classifier that inspects the first 100 rows
    of `text_col` and returns a tuple `(domain, language)`.

    domain   : one of 'clothing', 'shoes', 'electronics', or 'general'
    language : 'Indonesian' if Indonesian keyword hits exceed the total English
               domain-keyword hits; otherwise 'English'.

    The result is intended to drive ensemble routing — for now the same base
    pipeline is reused for every domain, but the structure leaves a clean
    extension point for per-domain models later.
    """
    if text_col not in df.columns or len(df) == 0:
        return "general", "English"

    # Take a small representative sample and lowercase it once.
    sample_series = df[text_col].head(100).fillna("").astype(str).str.lower()
    corpus = " ".join(sample_series.tolist())

    # --- Domain scoring (English keyword dictionaries) ---
    domain_counts: dict[str, int] = {}
    total_english_hits = 0
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = sum(corpus.count(kw) for kw in keywords)
        domain_counts[domain] = hits
        total_english_hits += hits

    # --- Language scoring (Indonesian keywords) ---
    indonesian_hits = sum(corpus.count(kw) for kw in INDONESIAN_KEYWORDS)
    language = "Indonesian" if indonesian_hits > total_english_hits else "English"

    # --- Pick winning domain (only if there is a non-zero match) ---
    best_domain, best_count = max(domain_counts.items(), key=lambda kv: kv[1])
    domain = best_domain if best_count > 0 else "general"

    return domain, language


# ============================================================================
# Rule-Based ML Correction — Accuracy Failsafe Layer
# ============================================================================
def apply_rule_based_correction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-process ML predictions with strict rating-based overrides.

    Dynamically locates a rating column (any column whose name contains
    'rating', 'score', or 'star', case-insensitively).

    Rule A — False Negative Fix:
        Predicted_IND == 0 AND rating >= 4   →   flip to Positive (1).
    Rule B — False Positive Fix:
        Predicted_IND == 1 AND rating <= 2   →   flip to Negative (0).

    If no rating column is found, the dataframe is returned untouched.
    Required columns: 'Predicted_IND' and 'Predicted_Sentiment'.
    """
    if "Predicted_IND" not in df.columns or "Predicted_Sentiment" not in df.columns:
        return df

    # Locate first column whose name contains rating/score/star.
    rating_col: Optional[str] = None
    for col in df.columns:
        name = str(col).strip().lower()
        if "rating" in name or "score" in name or "star" in name:
            rating_col = col
            break

    if rating_col is None:
        return df  # No rating signal available — leave predictions alone.

    corrected = df.copy()
    ratings = pd.to_numeric(corrected[rating_col], errors="coerce")

    # Rule A: model said negative but rating is clearly positive.
    mask_a = (corrected["Predicted_IND"] == NEGATIVE_LABEL) & (ratings >= 4)
    corrected.loc[mask_a, "Predicted_IND"] = POSITIVE_LABEL
    corrected.loc[mask_a, "Predicted_Sentiment"] = LABEL_MAP[POSITIVE_LABEL]

    # Rule B: model said positive but rating is clearly negative.
    mask_b = (corrected["Predicted_IND"] == POSITIVE_LABEL) & (ratings <= 2)
    corrected.loc[mask_b, "Predicted_IND"] = NEGATIVE_LABEL
    corrected.loc[mask_b, "Predicted_Sentiment"] = LABEL_MAP[NEGATIVE_LABEL]

    # Annotate corrections so the UI / downloads can audit them.
    corrected["Rule_Corrected"] = mask_a | mask_b

    return corrected


# ============================================================================
# Gemini Helpers
# ============================================================================
def get_gemini_api_key() -> Tuple[Optional[str], Optional[str]]:
    """Securely fetch Gemini API key. Returns (key, error_msg)."""
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if not key or not str(key).strip():
            return None, "GEMINI_API_KEY is empty in your secrets."
        return str(key).strip(), None
    except (KeyError, FileNotFoundError):
        return None, (
            "GEMINI_API_KEY not configured. Add it to `.streamlit/secrets.toml` "
            "or your deployment secrets to enable the AI Consultant."
        )
    except Exception as exc:
        return None, f"Could not read secrets: {exc}"


def call_gemini_consultant(
    api_key: str,
    negative_reviews: list[str],
    model_name: str = "gemini-2.5-flash",
    domain: str = "general",
    language: str = "English",
    rule_corrected_count: int = 0,
) -> str:
    """
    Send negative reviews to Gemini and request an aspect-based business
    intelligence summary dynamically using the selected model.

    The prompt adapts based on:
    - domain: adjusts suggested aspect categories (clothing vs shoes vs electronics)
    - language: if Indonesian, instructs Gemini to handle multilingual input
    - rule_corrected_count: informs Gemini about data pre-processing context
    """
    from google import genai
    client = genai.Client(api_key=api_key)

    joined = "\n".join(f"- {r}" for r in negative_reviews if str(r).strip())

    # --- Domain-specific aspect suggestions ---
    domain_aspects = {
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

    aspects_block = domain_aspects.get(domain, domain_aspects["general"])

    # --- Language instruction ---
    lang_instruction = ""
    if language == "Indonesian":
        lang_instruction = (
            "\n**IMPORTANT**: The reviews below are in Indonesian (Bahasa Indonesia). "
            "Analyze them in their original language but produce your report in English. "
            "Translate key phrases when quoting from reviews.\n\n"
        )

    # --- Rule correction context ---
    correction_note = ""
    if rule_corrected_count > 0:
        correction_note = (
            f"\n**Note**: A Rule-Based Correction system pre-filtered this data. "
            f"{rule_corrected_count} predictions were overridden where star ratings "
            f"strongly contradicted the ML model. The reviews below are confirmed "
            f"negatives after both ML and rule-based validation.\n\n"
        )

    prompt = (
        "You are a senior Business Consultant specializing in e-commerce "
        "customer experience analysis. Read the negative customer reviews "
        "below and produce an aspect-based business-intelligence report.\n\n"
        f"**Detected Domain**: {domain.title()}\n"
        f"**Detected Language**: {language}\n"
        f"{lang_instruction}"
        f"{correction_note}"
        "Categorize the pain points into specific business aspects such as "
        "(but not limited to):\n"
        f"{aspects_block}\n"
        "Only include aspects that actually appear in the reviews — skip any "
        "that don't have evidence. Quote short phrases from the reviews where "
        "useful.\n\n"
        "Format your response in clean Markdown with EXACTLY these sections "
        "and headings:\n\n"
        "## 📋 Executive Summary\n"
        "_2-3 sentences capturing the overall sentiment and the most "
        "significant systemic issues._\n\n"
        "## 🔍 Categorized Pain Points\n"
        "_For each relevant aspect, use a `### Aspect Name` sub-heading "
        "followed by a bulleted list of specific complaints. Keep bullets "
        "concise and concrete._\n\n"
        "## 🎯 High-Priority Action Items\n"
        "_A numbered list of 3-5 prioritized, actionable recommendations the "
        "business should implement next. Each item should reference which "
        "aspect(s) it addresses and the expected impact._\n\n"
        "---\n"
        "Negative Reviews:\n"
        f"{joined}\n"
        "---\n"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    return getattr(response, "text", str(response))


# ============================================================================
# UI — Premium SaaS Design (Indigo Light Mode)
# ============================================================================
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* === Base / Background === */
html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.stApp { background-color: #F8F9FA !important; }
[data-testid="stHeader"] { background: transparent !important; }

/* === Typography === */
h1, h2, h3, h4 { color: #111827 !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.2rem !important; }

/* === Hero header === */
.kiro-hero { padding: 4px 0 8px; margin-bottom: 18px; }
.kiro-hero-title {
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    color: #111827 !important;
}
.kiro-hero-sub {
    font-size: 0.95rem;
    color: #6B7280;
    margin: 0;
    line-height: 1.55;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 6px; }

.kiro-logo {
    display: flex;
    align-items: baseline;
    gap: 8px;
    padding: 18px 14px 4px 14px;
    color: #6366F1;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.kiro-logo-version {
    font-size: 0.7rem;
    color: #9CA3AF;
    font-weight: 500;
    letter-spacing: 0.2px;
}

.kiro-nav { display: flex; flex-direction: column; gap: 2px; padding: 6px 12px 12px; }
.kiro-nav a {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    color: #4B5563;
    text-decoration: none !important;
    font-weight: 500;
    font-size: 0.92rem;
    transition: all 0.15s ease;
}
.kiro-nav a:hover {
    background-color: #EEF2FF;
    color: #4F46E5;
}

.kiro-sidebar-card {
    margin: 12px;
    padding: 14px 16px;
    background-color: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
}
.kiro-sidebar-card-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: #111827;
    margin-bottom: 6px;
}
.kiro-sidebar-card-text {
    font-size: 0.78rem;
    color: #6B7280;
    line-height: 1.45;
    margin-bottom: 6px;
}

/* === Premium metric cards === */
.kiro-metric-card {
    background-color: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E5E7EB;
    padding: 18px 22px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.18s ease, box-shadow 0.18s ease;
    height: 100%;
}
.kiro-metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(79,70,229,0.10);
}
.kiro-metric-label {
    font-size: 0.78rem;
    color: #6B7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 6px;
}
.kiro-metric-value {
    font-size: 2.25rem;
    font-weight: 800;
    line-height: 1.15;
    color: #4F46E5;
}

/* === Buttons (primary indigo) === */
.stButton > button, .stDownloadButton > button {
    border-radius: 8px !important;
    padding: 0.55em 1.2em !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
    border: 1px solid transparent !important;
}
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    border: 1px solid #4F46E5 !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="baseButton-primary"]:hover {
    background-color: #4338CA !important;
    border-color: #4338CA !important;
    transform: translateY(-1px) scale(1.01);
    box-shadow: 0 6px 14px rgba(79,70,229,0.25) !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]) {
    background-color: #FFFFFF !important;
    color: #374151 !important;
    border: 1px solid #E5E7EB !important;
}
.stButton > button:not([kind="primary"]):not([data-testid="baseButton-primary"]):hover {
    background-color: #F3F4F6 !important;
    border-color: #D1D5DB !important;
}
.stDownloadButton > button {
    background-color: #FFFFFF !important;
    color: #4F46E5 !important;
    border: 1px solid #C7D2FE !important;
}
.stDownloadButton > button:hover {
    background-color: #EEF2FF !important;
    border-color: #A5B4FC !important;
}

/* === Alerts (info / warning / success / error) === */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    padding: 12px 16px !important;
    border: 1px solid !important;
    box-shadow: none !important;
}

/* === File uploader === */
[data-testid="stFileUploader"] section {
    border-radius: 10px !important;
    border: 1px dashed #C7D2FE !important;
    background-color: #F5F7FF !important;
}
[data-testid="stFileUploader"] button {
    background-color: #4F46E5 !important;
    color: #FFFFFF !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    border: none !important;
}
[data-testid="stFileUploader"] button:hover {
    background-color: #4338CA !important;
}

/* === Inputs === */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] > div > div,
[data-baseweb="select"] > div {
    border-radius: 8px !important;
    border-color: #E5E7EB !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.18) !important;
}

/* === Expander === */
[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #E5E7EB !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* === Divider === */
hr { border-color: #E5E7EB !important; opacity: 0.6 !important; }

/* === Dataframe === */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E5E7EB;
}

/* === Section anchors (invisible offset) === */
.kiro-anchor {
    display: block;
    position: relative;
    top: -50px;
    visibility: hidden;
}

/* === Footer === */
.kiro-footer {
    text-align: center;
    color: #9CA3AF;
    font-size: 0.8rem;
    padding: 24px 0 10px;
    border-top: 1px solid #E5E7EB;
    margin-top: 32px;
}
</style>
"""


def render_metric_html(label: str, value: str, color: str = "#4F46E5") -> str:
    """Render a premium metric card as HTML for st.markdown injection."""
    return (
        '<div class="kiro-metric-card">'
        f'<div class="kiro-metric-label">{label}</div>'
        f'<div class="kiro-metric-value" style="color: {color};">{value}</div>'
        '</div>'
    )


# --- Inject global CSS once ---
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# --- Hero header (with #dashboard anchor) ---
st.markdown(
    '<a id="dashboard" class="kiro-anchor"></a>'
    '<div class="kiro-hero">'
    '<div class="kiro-hero-title">🛍️ Hybrid E-Commerce Sentiment Analyzer</div>'
    '<p class="kiro-hero-sub">Traditional ML (TF-IDF + Logistic Regression) '
    '<strong>+</strong> Auto-Routing Domain Detection <strong>+</strong> '
    'Rule-Based ML Correction <strong>+</strong> Generative AI (Gemini) '
    'for actionable business insights.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# --- Train Base Model ---
try:
    base = train_base_pipeline()
except FileNotFoundError:
    st.error(
        f"Base dataset `{BASE_DATASET_PATH}` not found. "
        "Please place it in the project root."
    )
    st.stop()
except Exception as exc:
    st.error(f"Failed to train base model: {exc}")
    st.stop()

# --- Sidebar (premium SaaS layout) ---
with st.sidebar:
    # Brand logo + version
    st.markdown(
        '<div class="kiro-logo">📊 Market Insights'
        '<span class="kiro-logo-version">v2.4.0</span></div>',
        unsafe_allow_html=True,
    )

    # Navigation links (jump to section anchors)
    st.markdown(
        '<nav class="kiro-nav">'
        '<a href="#dashboard">🏠 Dashboard</a>'
        '<a href="#model-performance">📈 Model Performance</a>'
        '<a href="#sentiments">🔍 Sentiments</a>'
        '<a href="#ai-consultant">🤖 AI Consultant</a>'
        '</nav>',
        unsafe_allow_html=True,
    )

    # Upload card (visual frame around uploader)
    st.markdown(
        '<div class="kiro-sidebar-card">'
        '<div class="kiro-sidebar-card-title">📂 Upload CSV</div>'
        '<div class="kiro-sidebar-card-text">Upload a CSV with a text column. '
        "We&#39;ll auto-detect.</div></div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Drop a CSV file", type=["csv"], label_visibility="collapsed"
    )

    st.markdown('<hr/>', unsafe_allow_html=True)

    # Gemini status
    st.markdown('**🔑 Gemini Status**')
    api_key, api_err = get_gemini_api_key()
    if api_key:
        st.success("API key loaded.")
    else:
        st.warning(api_err)

    st.markdown('<hr/>', unsafe_allow_html=True)
    st.markdown(
        "**Tech Stack**\n"
        "- scikit-learn (TF-IDF + LogReg)\n"
        "- Pandas / Plotly\n"
        "- Google GenAI SDK"
    )

# --- Model Performance (dynamic — updated when domain pipeline trains) ---
# We use a placeholder approach: show base metrics initially, then override
# with domain-specific metrics if an uploaded dataset trains successfully.

# Initialize active_eval to the base pipeline stats (default view)
active_eval: dict = base

def render_model_performance(eval_data: dict) -> None:
    """Render the Model Performance section dynamically with premium cards."""
    domain_label = eval_data.get("domain", "clothing").title()
    st.markdown(
        '<a id="model-performance" class="kiro-anchor"></a>',
        unsafe_allow_html=True,
    )
    st.subheader("📈 Model Performance (Base Dataset + Hybrid Pipeline)")
    st.caption(
        "Base ML accuracy shown below. Uploaded data benefits from additional "
        "**Auto-Routing** (domain/language detection) and **Rule-Based Correction** "
        "(rating override) layers that improve effective accuracy beyond this baseline."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            render_metric_html(
                "Base Accuracy",
                f"{eval_data['accuracy'] * 100:.2f}%",
                color="#059669",  # Emerald green for accuracy
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            render_metric_html("Train Size", f"{eval_data['n_train']:,}"),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            render_metric_html("Test Size", f"{eval_data['n_test']:,}"),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            render_metric_html("Classes", str(len(eval_data["classes"]))),
            unsafe_allow_html=True,
        )

    with st.expander("📋 Classification Report & Pipeline Info", expanded=False):
        st.code(eval_data["report"], language="text")
        st.markdown(
            f"**Base Model**: TF-IDF (bigrams) + Logistic Regression "
            f"(**{domain_label}** Optimized)\n\n"
            f"**Enhancement Layers** (applied on user uploads):\n"
            f"1. 🔍 Auto-Routing — detects domain (clothing/shoes/electronics) & language (EN/ID)\n"
            f"2. ⚙️ Rule-Based Correction — overrides ML when star rating strongly disagrees\n"
            f"3. 🤖 Gemini AI — domain-aware prompt engineering for business insights"
        )

# Render initial (base) performance — will be overridden below if domain trains
_perf_placeholder = st.empty()
with _perf_placeholder.container():
    render_model_performance(active_eval)

st.divider()

# ============================================================================
# User Upload & Predictions  (with interactive filters)
# ============================================================================
st.markdown('<a id="sentiments" class="kiro-anchor"></a>', unsafe_allow_html=True)
st.subheader("🔍 Predict Sentiments on Your Data")

uploaded_df: Optional[pd.DataFrame] = None
predicted_df: Optional[pd.DataFrame] = None
user_text_col: Optional[str] = None

if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        uploaded_df = None

if uploaded_df is not None:
    user_text_col = find_text_column(uploaded_df)

    if user_text_col is None:
        # CRITICAL: exact error message required by spec
        st.error(
            "Could not detect a text column. Ensure your CSV has a column "
            "named 'Review Text', 'Review', 'Text', 'Comment', etc."
        )
    else:
        # ==================================================================
        # STEP 1: Auto-Routing — Detect domain & language via meta-classifier
        # ==================================================================
        detected_domain, detected_language = detect_dataset_domain(
            uploaded_df, user_text_col
        )

        # Display domain detection result
        st.info(f"🔍 Detected Dataset Domain: **{detected_domain.title()}**")

        # Display language detection result
        if detected_language == "Indonesian":
            st.info(
                "🇮🇩 Detected Language: Indonesian. "
                "Routing to multilingual handler."
            )

        # ==================================================================
        # STEP 1b: Train domain-specific pipeline if rating column exists
        # ==================================================================
        domain_eval: Optional[dict] = None
        with st.spinner(f"Training {detected_domain.title()} domain model..."):
            domain_eval = train_domain_pipeline(
                uploaded_df, user_text_col, detected_domain
            )

        if domain_eval is not None:
            # Domain pipeline trained successfully — use it for predictions
            active_pipeline = domain_eval["pipeline"]
            active_eval = domain_eval
            st.success(
                f"🎯 Domain-specific model trained on uploaded data! "
                f"Accuracy: **{domain_eval['accuracy'] * 100:.2f}%** "
                f"(Train: {domain_eval['n_train']:,} · Test: {domain_eval['n_test']:,})"
            )
        else:
            # Fallback to base pipeline (no rating column or insufficient data)
            active_pipeline = base["pipeline"]
            active_eval = base

        # --- Render dynamic Model Performance with active eval ---
        with _perf_placeholder.container():
            render_model_performance(active_eval)

        # ==================================================================
        # STEP 2: ML Prediction using the routed pipeline
        # ==================================================================
        # Clean text: fill NaN with empty string before predicting
        cleaned_texts = uploaded_df[user_text_col].fillna("").astype(str).tolist()

        with st.spinner("Predicting sentiments..."):
            preds = active_pipeline.predict(cleaned_texts)
            predicted_df = uploaded_df.copy()
            predicted_df["Predicted_IND"] = preds
            predicted_df["Predicted_Sentiment"] = [
                LABEL_MAP.get(int(p), "Unknown") for p in preds
            ]

        # ==================================================================
        # STEP 3: Rule-Based Correction — override ML where rating disagrees
        # ==================================================================
        predicted_df = apply_rule_based_correction(predicted_df)

        # Show correction stats if any corrections were applied
        if "Rule_Corrected" in predicted_df.columns:
            n_corrected = int(predicted_df["Rule_Corrected"].sum())
            if n_corrected > 0:
                st.warning(
                    f"⚙️ Rule-Based Correction applied to **{n_corrected:,}** "
                    f"rows where rating conflicted with ML prediction."
                )

        st.success(
            f"Predicted **{len(predicted_df):,}** rows using detected column "
            f"`{user_text_col}`."
        )

        # ----- Sentiment distribution chart -----
        dist = (
            predicted_df["Predicted_Sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )
        chart_col, table_col = st.columns([1, 1])

        with chart_col:
            st.markdown("**Sentiment Distribution**")
            fig = px.pie(
                dist, names="Sentiment", values="Count", hole=0.45,
                color="Sentiment",
                color_discrete_map={"Positive": "#22c55e", "Negative": "#ef4444"},
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with table_col:
            st.markdown("**Counts**")
            st.dataframe(dist, use_container_width=True, hide_index=True)
            st.bar_chart(dist.set_index("Sentiment")["Count"])

        st.divider()

        # ====================================================================
        # Product Intelligence — Dynamic Visualizations
        # ====================================================================
        st.subheader("📊 Product Intelligence")
        st.caption(
            "Automatic insights based on detected product and category columns. "
            "Charts adapt to your dataset structure."
        )

        # --- Dynamic column detection for product/category ---
        PRODUCT_CANDIDATES = ["product name", "title", "item", "product"]
        CATEGORY_CANDIDATES = ["category", "class name", "department name", "brand", "department"]

        def _find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
            """Find the first column whose lowercase name contains a candidate."""
            for col in df.columns:
                col_lower = str(col).strip().lower()
                for candidate in candidates:
                    if candidate in col_lower:
                        return col
            return None

        product_col = _find_column(predicted_df, PRODUCT_CANDIDATES)
        category_col = _find_column(predicted_df, CATEGORY_CANDIDATES)

        pi_left, pi_right = st.columns(2)

        # --- Insight 1: Top 5 Viral Products (Positive) ---
        with pi_left:
            st.markdown("**🌟 Top 5 Viral Products** (Most Positive Reviews)")
            if product_col is None:
                st.info(
                    "Product Name column not found in this dataset to "
                    "generate this insight."
                )
            else:
                positive_df = predicted_df[
                    predicted_df["Predicted_IND"] == POSITIVE_LABEL
                ].copy()
                positive_df[product_col] = (
                    positive_df[product_col].fillna("Unknown").astype(str)
                )
                top_products = (
                    positive_df.groupby(product_col)
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Count", ascending=False)
                    .head(5)
                )
                if top_products.empty:
                    st.info("No positive reviews found to rank products.")
                else:
                    fig_viral = px.bar(
                        top_products,
                        x="Count",
                        y=product_col,
                        orientation="h",
                        color_discrete_sequence=["#22c55e"],
                    )
                    fig_viral.update_layout(
                        showlegend=False,
                        yaxis_title="",
                        xaxis_title="Positive Review Count",
                        margin=dict(t=10, b=10, l=10, r=10),
                        yaxis=dict(autorange="reversed"),
                    )
                    st.plotly_chart(fig_viral, use_container_width=True)

        # --- Insight 2: Red Flag Categories (Negative) ---
        with pi_right:
            st.markdown("**🚨 Red Flag Categories** (Most Negative Reviews)")
            if category_col is None:
                st.info(
                    "Category/Brand column not found in this dataset to "
                    "generate this insight."
                )
            else:
                negative_df = predicted_df[
                    predicted_df["Predicted_IND"] == NEGATIVE_LABEL
                ].copy()
                negative_df[category_col] = (
                    negative_df[category_col].fillna("Unknown").astype(str)
                )
                cat_counts = (
                    negative_df.groupby(category_col)
                    .size()
                    .reset_index(name="Count")
                    .sort_values("Count", ascending=False)
                )
                if cat_counts.empty:
                    st.info("No negative reviews found to identify red flags.")
                else:
                    fig_flags = px.pie(
                        cat_counts,
                        names=category_col,
                        values="Count",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Reds_r,
                    )
                    fig_flags.update_traces(textinfo="percent+label")
                    fig_flags.update_layout(
                        showlegend=True,
                        margin=dict(t=10, b=10, l=10, r=10),
                    )
                    st.plotly_chart(fig_flags, use_container_width=True)

        st.divider()

        # ====================================================================
        # Interactive Predictions Dashboard (search + filter)
        # ====================================================================
        st.markdown("### 🗂️ Interactive Predictions Explorer")
        st.caption(
            "Search and filter all predicted reviews. Click any column header "
            "to sort."
        )

        filter_col1, filter_col2 = st.columns([2, 1])

        with filter_col1:
            search_keyword = st.text_input(
                "🔎 Search keyword in reviews",
                value="",
                placeholder="e.g. fabric, fit, late, returned ...",
                help="Case-insensitive substring search inside the review text.",
            )

        with filter_col2:
            sentiment_filter = st.radio(
                "Sentiment filter",
                options=["All", "Positive", "Negative"],
                horizontal=True,
                index=0,
            )

        # Apply filters to a working copy
        filtered_df = predicted_df.copy()

        if sentiment_filter != "All":
            filtered_df = filtered_df[
                filtered_df["Predicted_Sentiment"] == sentiment_filter
            ]

        if search_keyword.strip():
            keyword = search_keyword.strip().lower()
            mask = (
                filtered_df[user_text_col]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.contains(keyword, regex=False)
            )
            filtered_df = filtered_df[mask]

        # Total count of filtered rows ABOVE the table
        total_count = len(filtered_df)
        total_all = len(predicted_df)
        st.markdown(
            f"**Showing {total_count:,} of {total_all:,} reviews** "
            f"(sentiment: `{sentiment_filter}`"
            + (f", keyword: `{search_keyword}`" if search_keyword.strip() else "")
            + ")"
        )

        # Reorder columns: text first, then sentiment, then everything else
        display_cols = [user_text_col, "Predicted_Sentiment", "Predicted_IND"]
        other_cols = [c for c in filtered_df.columns if c not in display_cols]
        display_df = filtered_df[display_cols + other_cols].reset_index(drop=True)

        if total_count == 0:
            st.warning("No reviews match the current filters.")
        else:
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

        # ----- Download (full predicted data) -----
        csv_buf = io.StringIO()
        predicted_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download All Predictions (CSV)",
            data=csv_buf.getvalue(),
            file_name="predicted_reviews.csv",
            mime="text/csv",
        )
else:
    st.info(
        "👈 Upload a CSV from the sidebar to predict sentiments. "
        "The AI Consultant below can still analyze the base dataset's "
        "negative reviews."
    )

st.divider()

# Gemini model — using current stable, widely-available model on the Gemini
selected_model = st.selectbox(
    "Pilih Model AI Google (Free Tier):",
    ["gemini-2.5-flash", "gemini-3.1-flash-lite"],
    index=0
)

GEMINI_MODEL = selected_model
model_name = selected_model

# ============================================================================
# Gemini AI Consultant — Domain-Aware Aspect-Based Business Intelligence
# ============================================================================
st.markdown('<a id="ai-consultant" class="kiro-anchor"></a>', unsafe_allow_html=True)
st.subheader("🤖 Ask AI Consultant (Domain-Aware Aspect Analysis)")
st.markdown(
    "Generate an executive **aspect-based** report from negative reviews. "
    "The prompt automatically adapts to the detected **domain** (clothing/shoes/electronics) "
    "and **language** (English/Indonesian). Pain points are categorized by relevant "
    "business areas with prioritized action items."
)

trigger = st.button("🚀 Generate Insights", type="primary")


def collect_negative_samples() -> Tuple[list[str], str]:
    """Get up to 10 negative review texts for Gemini."""
    if predicted_df is not None and user_text_col is not None:
        neg_texts = [
            str(predicted_df.iloc[i][user_text_col])
            for i in range(len(predicted_df))
            if int(predicted_df.iloc[i]["Predicted_IND"]) == NEGATIVE_LABEL
            and str(predicted_df.iloc[i][user_text_col]).strip()
        ]
        if len(neg_texts) == 0:
            return [], "user_all_positive"
        return random.sample(neg_texts, min(10, len(neg_texts))), "your uploaded data"

    base_neg = [t for t in base.get("base_negative_samples", []) if str(t).strip()]
    if len(base_neg) == 0:
        return [], "no source"
    return random.sample(base_neg, k=min(10, len(base_neg))), "base dataset test set"


if trigger:
    if not api_key:
        st.warning(
            "Gemini API key not configured. Add `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml` then refresh."
        )
    else:
        samples, source = collect_negative_samples()
        if not samples:
            if source == "user_all_positive":
                st.success(
                    "Great news! There are no negative reviews in this "
                    "dataset. Your customers are fully satisfied!"
                )
            else:
                st.info("No negative reviews available to analyze.")
        else:
            # Determine domain/language context for prompt adaptation
            _gemini_domain = "general"
            _gemini_language = "English"
            _gemini_corrections = 0

            if predicted_df is not None and user_text_col is not None:
                # Use detected domain/language from the upload flow
                _gemini_domain, _gemini_language = detect_dataset_domain(
                    predicted_df, user_text_col
                )
                if "Rule_Corrected" in predicted_df.columns:
                    _gemini_corrections = int(predicted_df["Rule_Corrected"].sum())

            with st.expander(
                f"📝 Reviews sent to Gemini ({source})", expanded=False
            ):
                for i, s in enumerate(samples, 1):
                    st.markdown(f"{i}. {s}")
                st.caption(
                    f"Context → Domain: **{_gemini_domain.title()}** · "
                    f"Language: **{_gemini_language}** · "
                    f"Rule Corrections: **{_gemini_corrections}**"
                )
            with st.spinner(f"Consulting {GEMINI_MODEL}..."):
                try:
                    output = call_gemini_consultant(
                        api_key,
                        samples,
                        model_name=GEMINI_MODEL,
                        domain=_gemini_domain,
                        language=_gemini_language,
                        rule_corrected_count=_gemini_corrections,
                    )
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
                else:
                    st.markdown("### 💼 Business Intelligence Report")
                    st.markdown(output)

# --- Footer ---
st.markdown(
    '<div class="kiro-footer">Built with Streamlit · scikit-learn · '
    'Google GenAI SDK · Hybrid ML + GenAI architecture.</div>',
    unsafe_allow_html=True,
)
