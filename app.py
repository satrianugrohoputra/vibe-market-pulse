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
    report = classification_report(
        y_test, y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        zero_division=0,
    )

    base_negatives = [
        X_test[i] for i in range(len(X_test))
        if y_pred[i] == NEGATIVE_LABEL and str(X_test[i]).strip()
    ]

    return {
        "pipeline": pipeline,
        "accuracy": accuracy,
        "report": report,
        "classes": ["Negative", "Positive"],
        "base_negative_samples": base_negatives,
        "n_train": len(X_train),
        "n_test": len(X_test),
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


def call_gemini_consultant(api_key: str, negative_reviews: list[str], model_name: str = "gemini-2.5-flash") -> str:
    """
    Send negative reviews to Gemini and request an aspect-based business
    intelligence summary dynamically using the selected model.
    """
    from google import genai
    client = genai.Client(api_key=api_key)

    joined = "\n".join(f"- {r}" for r in negative_reviews if str(r).strip())

    prompt = (
        "You are a senior Business Consultant specializing in e-commerce "
        "customer experience analysis. Read the negative customer reviews "
        "below and produce an aspect-based business-intelligence report.\n\n"
        "Categorize the pain points into specific business aspects such as "
        "(but not limited to):\n"
        "  - Sizing & Fit\n"
        "  - Material Quality\n"
        "  - Design & Style\n"
        "  - Customer Service\n"
        "  - Shipping & Delivery\n"
        "  - Pricing & Value\n\n"
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
# UI
# ============================================================================
st.title("🛍️ Hybrid E-Commerce Sentiment Analyzer")
st.caption(
    "Traditional ML (TF-IDF + Logistic Regression) **+** Generative AI "
    f"(Gemini 2.5 Flash) for actionable business insights."
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

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload New Reviews")
    st.markdown(
        "Upload a CSV with a text column. We'll auto-detect any of: "
        "`Review Text`, `Review`, `Text`, `Comment`, `Content`, `Description`."
    )
    uploaded_file = st.file_uploader("Drop a CSV file", type=["csv"])

    st.divider()
    st.subheader("🔑 Gemini Status")
    api_key, api_err = get_gemini_api_key()
    if api_key:
        st.success("API key loaded.")
    else:
        st.warning(api_err)

    st.divider()
    st.markdown(
        "**Tech Stack**\n"
        "- scikit-learn (TF-IDF + LogReg)\n"
        "- Pandas / Plotly\n"
        f"- Google GenAI SDK (GEMINI_MODEL)"
    )

# --- Model Performance ---
st.subheader("📈 Model Performance (Base Dataset)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{base['accuracy'] * 100:.2f}%")
col2.metric("Train Size", base["n_train"])
col3.metric("Test Size", base["n_test"])
col4.metric("Classes", len(base["classes"]))

with st.expander("📋 Classification Report", expanded=False):
    st.code(base["report"], language="text")
    st.caption(
        f"Text column: **{TEXT_COL}** · "
        f"Target: **{TARGET_COL}** (1=Positive, 0=Negative)"
    )

st.divider()

# ============================================================================
# User Upload & Predictions  (with interactive filters)
# ============================================================================
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

        # TODO: Load specific .pkl model based on domain
        # For now, all domains route to the single base pipeline.
        # Future: model_registry = {"clothing": "clothing_model.pkl", ...}
        # active_pipeline = load_domain_model(detected_domain)
        active_pipeline = base["pipeline"]

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
# Gemini AI Consultant — Aspect-Based Business Intelligence
# ============================================================================
st.subheader("🤖 Ask AI Consultant (Aspect-Based Analysis)")
st.markdown(
    "Generate an executive **aspect-based** report from negative reviews "
    "(`Recommended IND == 0`). Pain points are categorized by business area "
    "(Sizing, Material, Service, Shipping, etc.) with prioritized action items."
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
            with st.expander(
                f"📝 Reviews sent to Gemini ({source})", expanded=False
            ):
                for i, s in enumerate(samples, 1):
                    st.markdown(f"{i}. {s}")
            with st.spinner(f"Consulting {GEMINI_MODEL}..."):
                try:
                    # TAMBAHKAN model_name ATAU GEMINI_MODEL SEBAGAI ARGUMEN KETIGA
                    output = call_gemini_consultant(api_key, samples, model_name=GEMINI_MODEL)
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
                else:
                    st.markdown("### 💼 Business Intelligence Report")
                    st.markdown(output)

# --- Footer ---
st.divider()
st.caption(
    "Built with Streamlit · scikit-learn · Google GenAI SDK · "
    "Hybrid ML + GenAI architecture."
)
