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

# Numeric label conventions for "Recommended IND"
NEGATIVE_LABEL = 0  # not recommended
POSITIVE_LABEL = 1  # recommended

# Friendly display names for the two classes
LABEL_DISPLAY = {
    NEGATIVE_LABEL: "Negative",
    POSITIVE_LABEL: "Positive",
}

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
# Data cleaning helpers
# ----------------------------------------------------------------------------
def clean_review_dataframe(
    df: pd.DataFrame,
    text_col: str = TEXT_COL,
    target_col: Optional[str] = TARGET_COL,
) -> pd.DataFrame:
    """
    Clean a reviews dataframe so it is safe to feed into TfidfVectorizer.

    Rules:
      * Drop rows where the target column is NaN (only when target_col is given).
      * Fill NaN in the text column with empty string ('').
      * Coerce the target column to int (0 / 1).
    """
    out = df.copy()

    if target_col is not None and target_col in out.columns:
        # Drop rows where the target/label is missing - these are unusable for
        # supervised training.
        out = out.dropna(subset=[target_col])
        # Force numeric, drop anything that won't coerce, then cast to int.
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
        out = out.dropna(subset=[target_col])
        out[target_col] = out[target_col].astype(int)

    if text_col in out.columns:
        # CRITICAL: TfidfVectorizer will raise AttributeError on NaN in the text
        # column, so fill with empty string before passing it any text.
        out[text_col] = out[text_col].fillna("").astype(str)

    return out


def find_text_column(df: pd.DataFrame) -> Optional[str]:
    """
    Locate a usable review text column on a user-uploaded CSV.

    Priority:
      1. The exact base column name ("Review Text").
      2. Common alternates (Review, Text, Comment, ...).
      3. Fallback: the object column with the longest average string length.
    """
    if TEXT_COL in df.columns:
        return TEXT_COL

    candidates = [
        "Review Text", "review_text", "Review", "review",
        "Text", "text", "Reviews", "reviews",
        "Comment", "comment", "Feedback", "feedback",
        "Message", "message", "Content", "content", "Body", "body",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    object_cols = df.select_dtypes(include="object").columns.tolist()
    if not object_cols:
        return None
    avg_lengths = {c: df[c].fillna("").astype(str).str.len().mean() for c in object_cols}
    return max(avg_lengths, key=avg_lengths.get) if avg_lengths else None


# ----------------------------------------------------------------------------
# Cached: Train Base ML Pipeline
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training base sentiment model...")
def train_base_pipeline() -> dict:
    """
    Loads the base ecommercereviews.csv, trains a TF-IDF + Logistic Regression
    pipeline (80/20 split) on `Review Text` -> `Recommended IND`, and returns
    the trained pipeline plus evaluation artifacts.
    """
    df = pd.read_csv(BASE_DATASET_PATH)

    # Validate the expected columns exist
    missing = [c for c in (TEXT_COL, TARGET_COL) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Base dataset is missing required column(s) {missing}. "
            f"Found: {list(df.columns)}"
        )

    # CRITICAL data cleaning: drop NaN labels, fill NaN text with ''.
    df = clean_review_dataframe(df, text_col=TEXT_COL, target_col=TARGET_COL)

    # Keep only rows that are valid for our binary classification.
    df = df[df[TARGET_COL].isin([NEGATIVE_LABEL, POSITIVE_LABEL])]

    if df.empty:
        raise ValueError("No usable rows after cleaning the base dataset.")

    X = df[TEXT_COL].values
    y = df[TARGET_COL].values

    # Stratify only when both classes have at least 2 samples.
    stratify = y if pd.Series(y).value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify,
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
    target_names = [LABEL_DISPLAY[c] for c in sorted(set(y_test))]
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )

    # Sample of negatives from the held-out test set, used by the AI Consultant
    # when no user file has been uploaded.
    test_df = pd.DataFrame({"text": X_test, "actual": y_test, "predicted": y_pred})
    base_negatives = test_df.loc[
        test_df["predicted"] == NEGATIVE_LABEL, "text"
    ].astype(str).tolist()

    return {
        "pipeline": pipeline,
        "text_col": TEXT_COL,
        "target_col": TARGET_COL,
        "accuracy": accuracy,
        "report": report,
        "classes": [LABEL_DISPLAY[c] for c in pipeline.classes_],
        "base_negative_samples": base_negatives,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


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
        "Upload a CSV containing a `Review Text` column. "
        "The trained model will predict whether each review is "
        "**Positive** (recommended) or **Negative** (not recommended)."
    )
    uploaded_file = st.file_uploader(
        "Drop a CSV file",
        type=["csv"],
        help="Should include a column named 'Review Text' (or similar).",
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
        f"Text column: **{base['text_col']}** · "
        f"Target column: **{base['target_col']}** "
        f"(1 = Positive / Recommended, 0 = Negative / Not Recommended)"
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
    user_text_col = find_text_column(uploaded_df)
    if user_text_col is None:
        st.error(
            "Could not detect a text column in your file. Please ensure your "
            "CSV has a column named `Review Text` (or `Review`, `Text`, `Comment`, ...)."
        )
    else:
        # Apply the same cleaning rules used during training, so the
        # vectorizer never receives NaN.
        cleaned_df = clean_review_dataframe(
            uploaded_df, text_col=user_text_col, target_col=None
        )

        with st.spinner("Predicting sentiments..."):
            preds = base["pipeline"].predict(cleaned_df[user_text_col].values)
            predicted_df = cleaned_df.copy()
            predicted_df["Predicted_IND"] = preds
            predicted_df["Predicted_Sentiment"] = predicted_df["Predicted_IND"].map(
                LABEL_DISPLAY
            )

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
        preview_cols = [user_text_col, "Predicted_IND", "Predicted_Sentiment"]
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
    "and concrete recommendations, generated from negative reviews "
    "(`Recommended IND == 0`)."
)

ai_col1, _ = st.columns([1, 3])
trigger = ai_col1.button(
    "🚀 Generate Insights", type="primary", use_container_width=True
)


def collect_negative_samples() -> Tuple[list[str], str]:
    """
    Filter for negative reviews (Recommended IND == 0) and return up to 10
    randomly-sampled review texts for Gemini.

    Preference order:
      1. Predicted negatives from the user-uploaded dataset.
      2. Predicted negatives from the base dataset's held-out test set.
    """
    if predicted_df is not None and user_text_col is not None:
        neg_mask = predicted_df["Predicted_IND"] == NEGATIVE_LABEL
        neg_texts = (
            predicted_df.loc[neg_mask, user_text_col]
            .fillna("")
            .astype(str)
            .tolist()
        )
        neg_texts = [t for t in neg_texts if t.strip()]
        if neg_texts:
            sample = random.sample(neg_texts, k=min(10, len(neg_texts)))
            return sample, "your uploaded dataset"

    base_neg = [t for t in (base.get("base_negative_samples") or []) if str(t).strip()]
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
            with st.expander(
                f"📝 Reviews sent to Gemini (from {source})", expanded=False
            ):
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
