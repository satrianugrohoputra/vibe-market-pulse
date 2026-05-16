"""
Hybrid E-Commerce Sentiment Analyzer
====================================
A Streamlit app that combines a Traditional NLP ML pipeline (TF-IDF + Logistic
Regression) with Generative AI (Gemini 1.5 Pro) for executive-level insights
on e-commerce review datasets.

Base dataset columns (Women's E-Commerce Clothing Reviews):
    - "Review Text"     : free-text customer review
    - "Recommended IND" : 1 = Positive (recommended), 0 = Negative (not recommended)
"""

from __future__ import annotations

import io
import random
from typing import Optional, Tuple

import numpy as np
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
TEXT_COL = "Review Text"
TARGET_COL = "Recommended IND"

NEGATIVE_LABEL = 0  # not recommended
POSITIVE_LABEL = 1  # recommended

LABEL_MAP = {0: "Negative", 1: "Positive"}

# ----------------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid E-Commerce Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----------------------------------------------------------------------------
# Cached: Train Base ML Pipeline
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training base sentiment model...")
def train_base_pipeline() -> dict:
    """
    Loads the base CSV, cleans NaN values, trains a TF-IDF + Logistic Regression
    pipeline (80/20 split) and returns the trained pipeline plus evaluation info.
    """
    # --- Load ---
    df = pd.read_csv(BASE_DATASET_PATH)

    # --- Validate columns exist ---
    if TEXT_COL not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COL}' not found. Available: {list(df.columns)}"
        )
    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Column '{TARGET_COL}' not found. Available: {list(df.columns)}"
        )

    # --- CRITICAL: Handle missing values ---
    # 1. Drop rows where target (Recommended IND) is NaN
    df = df.dropna(subset=[TARGET_COL])

    # 2. Fill NaN in Review Text with empty string (prevents AttributeError
    #    in TfidfVectorizer which calls .lower() on each element)
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    # 3. Ensure correct dtypes
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df[TEXT_COL] = df[TEXT_COL].astype(str)

    # 4. Keep only valid binary labels
    df = df[df[TARGET_COL].isin([0, 1])].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No usable rows after cleaning.")

    # --- Prepare arrays (convert to Python lists to avoid numpy indexing issues) ---
    X = df[TEXT_COL].tolist()
    y = df[TARGET_COL].tolist()

    # --- Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    # --- Build & Train Pipeline ---
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

    # --- Evaluate ---
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["Negative (0)", "Positive (1)"],
        zero_division=0,
    )

    # --- Collect base negative samples for Gemini fallback ---
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


# ----------------------------------------------------------------------------
# Helper: find text column in user-uploaded CSV
# ----------------------------------------------------------------------------
def find_text_column(df: pd.DataFrame) -> Optional[str]:
    """Locate a usable text column in a user-uploaded CSV."""
    candidates = [
        "Review Text", "review_text", "Review", "review",
        "Text", "text", "Reviews", "reviews",
        "Comment", "comment", "Feedback", "feedback",
        "Message", "message", "Content", "content", "Body", "body",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: longest average string column
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        return None
    avg_len = {c: df[c].fillna("").astype(str).str.len().mean() for c in obj_cols}
    return max(avg_len, key=avg_len.get)


# ----------------------------------------------------------------------------
# Gemini Helpers
# ----------------------------------------------------------------------------
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


def call_gemini_consultant(api_key: str, negative_reviews: list[str]) -> str:
    """Send negative reviews to Gemini 1.5 Pro and return markdown response."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-pro")

    joined = "\n".join(f"- {r}" for r in negative_reviews if str(r).strip())

    prompt = (
        "Act as a Business Consultant. Read these negative customer reviews "
        "from an e-commerce store. Provide a concise executive summary of the "
        "core pain points and suggest 3 actionable business improvements.\n\n"
        "Format your response in clean Markdown with the following sections:\n"
        "1. **Executive Summary** (2-3 sentences)\n"
        "2. **Core Pain Points** (bullet list)\n"
        "3. **3 Actionable Recommendations** (numbered list, each with a short rationale)\n\n"
        "Negative Reviews:\n"
        f"{joined}"
    )

    response = model.generate_content(prompt)
    return getattr(response, "text", str(response))


# ============================================================================
# UI
# ============================================================================

st.title("🛍️ Hybrid E-Commerce Sentiment Analyzer")
st.caption(
    "Traditional ML (TF-IDF + Logistic Regression) **+** Generative AI "
    "(Gemini 1.5 Pro) for actionable business insights."
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
        "Upload a CSV with a **Review Text** column. "
        "The trained model will predict Positive or Negative for each review."
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
        "- Google Generative AI (Gemini 1.5 Pro)"
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

# --- User Upload & Predictions ---
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
        st.error(
            "Could not detect a text column. Ensure your CSV has a column named "
            "'Review Text', 'Review', 'Text', 'Comment', etc."
        )
    else:
        # Clean: fill NaN text with empty string
        cleaned_texts = uploaded_df[user_text_col].fillna("").astype(str).tolist()

        with st.spinner("Predicting sentiments..."):
            preds = base["pipeline"].predict(cleaned_texts)
            predicted_df = uploaded_df.copy()
            predicted_df["Predicted_IND"] = preds
            predicted_df["Predicted_Sentiment"] = [
                LABEL_MAP.get(int(p), "Unknown") for p in preds
            ]

        st.success(
            f"Predicted **{len(predicted_df)}** rows using column `{user_text_col}`."
        )

        # Pie chart
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

        st.markdown("**First 10 Predictions**")
        st.dataframe(
            predicted_df[[user_text_col, "Predicted_IND", "Predicted_Sentiment"]].head(10),
            use_container_width=True, hide_index=True,
        )

        # Download
        csv_buf = io.StringIO()
        predicted_df.to_csv(csv_buf, index=False)
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_buf.getvalue(),
            file_name="predicted_reviews.csv",
            mime="text/csv",
        )
else:
    st.info(
        "👈 Upload a CSV from the sidebar to predict sentiments. "
        "The AI Consultant below can still analyze the base dataset's negative reviews."
    )

st.divider()

# --- Gemini AI Consultant ---
st.subheader("🤖 Ask AI Consultant (Gemini 1.5 Pro)")
st.markdown(
    "Generate an executive summary of customer pain points from negative reviews "
    "(`Recommended IND == 0`)."
)

trigger = st.button("🚀 Generate Insights", type="primary")


def collect_negative_samples() -> Tuple[list[str], str]:
    """Get up to 10 negative review texts for Gemini."""
    # Try user upload first
    if predicted_df is not None and user_text_col is not None:
        neg_texts = [
            str(predicted_df.iloc[i][user_text_col])
            for i in range(len(predicted_df))
            if int(predicted_df.iloc[i]["Predicted_IND"]) == NEGATIVE_LABEL
            and str(predicted_df.iloc[i][user_text_col]).strip()
        ]
        if neg_texts:
            return random.sample(neg_texts, k=min(10, len(neg_texts))), "your uploaded data"

    # Fallback: base dataset negatives
    base_neg = [t for t in base.get("base_negative_samples", []) if str(t).strip()]
    if base_neg:
        return random.sample(base_neg, k=min(10, len(base_neg))), "base dataset test set"
    return [], "no source"


if trigger:
    if not api_key:
        st.warning(
            "Gemini API key not configured. Add `GEMINI_API_KEY` to "
            "`.streamlit/secrets.toml` then refresh."
        )
    else:
        samples, source = collect_negative_samples()
        if not samples:
            st.info("No negative reviews available to analyze.")
        else:
            with st.expander(f"📝 Reviews sent to Gemini ({source})", expanded=False):
                for i, s in enumerate(samples, 1):
                    st.markdown(f"{i}. {s}")
            with st.spinner("Consulting Gemini 1.5 Pro..."):
                try:
                    output = call_gemini_consultant(api_key, samples)
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
                else:
                    st.markdown("### 💼 Executive Insights")
                    st.markdown(output)

# --- Footer ---
st.divider()
st.caption(
    "Built with Streamlit · scikit-learn · Google Generative AI · "
    "Hybrid ML + GenAI architecture."
)
