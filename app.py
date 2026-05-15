"""
Hybrid E-Commerce Sentiment Analyzer
====================================
A Streamlit app that combines a Traditional NLP ML pipeline (TF-IDF + Logistic
Regression) with Generative AI (Gemini 1.5 Pro) for executive-level insights
on user-uploaded e-commerce review datasets.
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
# Page Config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Hybrid E-Commerce Sentiment Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DATASET_PATH = "ecommercereviews.csv"

# Common candidate column names for auto-detection
TEXT_COLUMN_CANDIDATES = [
    "Review", "review", "Text", "text", "Reviews", "reviews",
    "Comment", "comment", "Feedback", "feedback", "Message", "message",
    "Content", "content", "Body", "body",
]
TARGET_COLUMN_CANDIDATES = [
    "Sentiment", "sentiment", "Label", "label", "Rating", "rating",
    "Class", "class", "Target", "target", "Score", "score",
]

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def detect_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first matching column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely text column by name, falling back to longest avg string."""
    direct = detect_column(df, TEXT_COLUMN_CANDIDATES)
    if direct is not None:
        return direct
    # Fallback: pick the object column with the longest average string length
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if not object_cols:
        return None
    avg_lengths = {c: df[c].astype(str).str.len().mean() for c in object_cols}
    return max(avg_lengths, key=avg_lengths.get) if avg_lengths else None


def detect_target_column(df: pd.DataFrame, exclude: Optional[str] = None) -> Optional[str]:
    """Detect a target/sentiment column."""
    direct = detect_column(df, TARGET_COLUMN_CANDIDATES)
    if direct is not None and direct != exclude:
        return direct
    # Fallback: any column with a small number of unique values that isn't the text col
    candidates = [
        c for c in df.columns
        if c != exclude and df[c].nunique(dropna=True) <= 10
    ]
    return candidates[0] if candidates else None


# ----------------------------------------------------------------------------
# Cached: Train Base ML Pipeline
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training base sentiment model...")
def train_base_pipeline() -> dict:
    """
    Loads the base ecommercereviews.csv, auto-detects columns, trains a
    TF-IDF + Logistic Regression pipeline on an 80/20 split, and returns
    the trained pipeline along with evaluation artifacts.
    """
    df = pd.read_csv(BASE_DATASET_PATH)

    text_col = detect_text_column(df)
    target_col = detect_target_column(df, exclude=text_col)

    if text_col is None or target_col is None:
        raise ValueError(
            f"Could not detect text/target columns in {BASE_DATASET_PATH}. "
            f"Found columns: {list(df.columns)}"
        )

    # Clean & drop NA rows
    df = df[[text_col, target_col]].dropna()
    df[text_col] = df[text_col].astype(str)
    df[target_col] = df[target_col].astype(str)

    X = df[text_col].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42,
        stratify=y if pd.Series(y).value_counts().min() >= 2 else None,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
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
    report = classification_report(y_test, y_pred, zero_division=0)

    # Build a small "negative test set" for fallback Gemini analysis
    test_df = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": y_pred})
    negative_label = _resolve_negative_label(pd.Series(y))
    base_negatives = (
        test_df[test_df["predicted"].astype(str).str.lower() == str(negative_label).lower()]
        ["text"].tolist()
        if negative_label is not None else []
    )

    return {
        "pipeline": pipeline,
        "text_col": text_col,
        "target_col": target_col,
        "accuracy": accuracy,
        "report": report,
        "classes": list(pipeline.classes_),
        "base_negative_samples": base_negatives,
        "negative_label": negative_label,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def _resolve_negative_label(y: pd.Series) -> Optional[str]:
    """Find the label that represents 'negative' sentiment."""
    unique_labels = [str(u) for u in y.unique()]
    # Try common negative label names
    for candidate in ["Negative", "negative", "NEGATIVE", "neg", "0", "1"]:
        if candidate in unique_labels:
            # For numeric ratings, "1" is typically the lowest/negative
            if candidate in ("0", "1"):
                # Only treat numeric as negative if labels look like ratings
                if all(u.isdigit() for u in unique_labels):
                    return candidate
                continue
            return candidate
    return None


# ----------------------------------------------------------------------------
# Gemini Helpers
# ----------------------------------------------------------------------------
def get_gemini_api_key() -> Tuple[Optional[str], Optional[str]]:
    """
    Securely fetch the Gemini API key from st.secrets.
    Returns (api_key, error_message). Never raises.
    """
    try:
        key = st.secrets["GEMINI_API_KEY"]
        if not key or not str(key).strip():
            return None, "GEMINI_API_KEY is empty in your secrets."
        return str(key), None
    except (KeyError, FileNotFoundError):
        return None, (
            "GEMINI_API_KEY not configured. Add it to `.streamlit/secrets.toml` "
            "or your deployment secrets to enable the AI Consultant."
        )
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"Could not read secrets: {exc}"


def call_gemini_consultant(api_key: str, negative_reviews: list[str]) -> str:
    """Send negative reviews to Gemini 1.5 Pro and return the markdown response."""
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
    # google-generativeai returns .text for simple prompts
    return getattr(response, "text", str(response))


# ----------------------------------------------------------------------------
# UI: Header
# ----------------------------------------------------------------------------
st.title("🛍️ Hybrid E-Commerce Sentiment Analyzer")
st.caption(
    "Traditional ML (TF-IDF + Logistic Regression) **+** Generative AI "
    "(Gemini 1.5 Pro) for actionable business insights."
)

# ----------------------------------------------------------------------------
# Train Base Model (always runs, cached)
# ----------------------------------------------------------------------------
try:
    base = train_base_pipeline()
except FileNotFoundError:
    st.error(
        f"Base dataset `{BASE_DATASET_PATH}` not found. Please place it in the "
        "project root before running the app."
    )
    st.stop()
except Exception as exc:
    st.error(f"Failed to train base model: {exc}")
    st.stop()

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("📂 Upload New Reviews")
    st.markdown(
        "Upload a CSV containing a column of customer review text. "
        "The trained model will predict each review's sentiment."
    )
    uploaded_file = st.file_uploader(
        "Drop a CSV file",
        type=["csv"],
        help="Should include a column with review text (e.g., 'Review', 'Text', 'Comment').",
    )

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

# ----------------------------------------------------------------------------
# Section: Base Model Performance
# ----------------------------------------------------------------------------
st.subheader("📈 Model Performance (Base Dataset)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{base['accuracy'] * 100:.2f}%")
col2.metric("Train Size", base["n_train"])
col3.metric("Test Size", base["n_test"])
col4.metric("Classes", len(base["classes"]))

with st.expander("📋 Classification Report (held-out test set)", expanded=False):
    st.code(base["report"], language="text")
    st.caption(
        f"Detected text column: **{base['text_col']}** · "
        f"Detected target column: **{base['target_col']}** · "
        f"Classes: {', '.join(base['classes'])}"
    )

st.divider()

# ----------------------------------------------------------------------------
# Section: User Uploaded Predictions
# ----------------------------------------------------------------------------
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
    user_text_col = detect_text_column(uploaded_df)
    if user_text_col is None:
        st.error(
            "Could not detect a text column in your file. Please ensure your "
            "CSV has a column named one of: "
            f"{', '.join(TEXT_COLUMN_CANDIDATES[:6])}, ..."
        )
    else:
        with st.spinner("Predicting sentiments..."):
            texts = uploaded_df[user_text_col].astype(str).fillna("")
            preds = base["pipeline"].predict(texts.values)
            predicted_df = uploaded_df.copy()
            predicted_df["Predicted_Sentiment"] = preds

        st.success(
            f"Predicted **{len(predicted_df)}** rows using column `{user_text_col}`."
        )

        # Distribution chart
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
                dist,
                names="Sentiment",
                values="Count",
                hole=0.45,
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#22c55e",
                    "Negative": "#ef4444",
                    "Neutral": "#94a3b8",
                },
            )
            fig.update_traces(textinfo="percent+label")
            fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

        with table_col:
            st.markdown("**Counts**")
            st.dataframe(dist, use_container_width=True, hide_index=True)
            st.bar_chart(dist.set_index("Sentiment")["Count"])

        st.markdown("**First 10 Predictions**")
        preview_cols = [user_text_col, "Predicted_Sentiment"]
        st.dataframe(
            predicted_df[preview_cols].head(10),
            use_container_width=True,
            hide_index=True,
        )

        # Download button
        csv_buffer = io.StringIO()
        predicted_df.to_csv(csv_buffer, index=False)
        st.download_button(
            "⬇️ Download Predictions CSV",
            data=csv_buffer.getvalue(),
            file_name="predicted_reviews.csv",
            mime="text/csv",
        )
else:
    st.info(
        "👈 Upload a CSV from the sidebar to predict sentiments on your own data. "
        "If you skip this, the AI Consultant below will analyze the base dataset's "
        "negative test reviews instead."
    )

st.divider()

# ----------------------------------------------------------------------------
# Section: Gemini Executive Summary
# ----------------------------------------------------------------------------
st.subheader("🤖 Ask AI Consultant (Gemini 1.5 Pro)")
st.markdown(
    "Get a **business-level executive summary** of customer pain points "
    "and concrete recommendations, generated from negative reviews."
)

ai_col1, ai_col2 = st.columns([1, 3])
trigger = ai_col1.button("🚀 Generate Insights", type="primary", use_container_width=True)

# Decide which negative reviews to send
def collect_negative_samples() -> Tuple[list[str], str]:
    """Return (samples, source_label)."""
    if predicted_df is not None and user_text_col is not None:
        neg_mask = (
            predicted_df["Predicted_Sentiment"].astype(str).str.lower() == "negative"
        )
        neg_texts = predicted_df.loc[neg_mask, user_text_col].astype(str).tolist()
        if neg_texts:
            sample = random.sample(neg_texts, k=min(10, len(neg_texts)))
            return sample, "your uploaded dataset"
    # Fallback to base test negatives
    base_neg = base.get("base_negative_samples") or []
    if base_neg:
        sample = random.sample(base_neg, k=min(10, len(base_neg)))
        return sample, "the base dataset's negative test reviews"
    return [], "no source"


if trigger:
    if not api_key:
        st.warning(
            "Gemini API key is not configured. Add `GEMINI_API_KEY` to your "
            "`.streamlit/secrets.toml` (locally) or to your Streamlit Cloud "
            "secrets, then refresh."
        )
    else:
        samples, source = collect_negative_samples()
        if not samples:
            st.info("No negative reviews available to analyze.")
        else:
            with st.expander(f"📝 Reviews sent to Gemini (from {source})", expanded=False):
                for i, s in enumerate(samples, start=1):
                    st.markdown(f"{i}. {s}")
            with st.spinner("Consulting Gemini 1.5 Pro..."):
                try:
                    output = call_gemini_consultant(api_key, samples)
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
                else:
                    st.markdown("### 💼 Executive Insights")
                    st.markdown(output)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit · scikit-learn · Google Generative AI · "
    "Hybrid ML + GenAI architecture."
)
