# Vibe Market Pulse

A hybrid **e-commerce review analyzer** that turns raw customer reviews into
clear business insights. Upload a CSV of product reviews and the app predicts
sentiment, visualizes trends, and generates an executive summary of customer
complaints using AI.

> **Goal:** help a seller or analyst quickly understand *what customers love*
> and *what they complain about* — without reading thousands of reviews by hand.

---

## What it does

1. **Predicts sentiment** of every review (Positive / Negative) using a
   classic machine-learning model (TF-IDF + Logistic Regression).
2. **Auto-detects** the dataset's domain (clothing / shoes / electronics) and
   language (English / Indonesian).
3. **Corrects** obvious mistakes by cross-checking the star rating against the
   predicted sentiment.
4. **Visualizes** results: sentiment distribution, top "viral" products, and
   "red flag" categories.
5. **Smart Semantic Search** — find reviews by *meaning*, not just keywords
   (works across English and Indonesian).
6. **AI Consultant (Agentic RAG)** — a LangGraph workflow that retrieves the
   most relevant negative reviews, groups them into themes, and asks Google
   Gemini to write a short business report with action items.

---

## How it works (in short)

```
Upload CSV
   -> read it safely (any encoding / delimiter)
   -> detect text column, domain, language
   -> train a sentiment model on your data
   -> predict sentiment + confidence
   -> build a semantic index (embeddings)
        |-> Smart Search (find by meaning)
        |-> AI Consultant (LangGraph -> Gemini report)
```

The app combines **traditional ML** (fast, free, explainable) with
**Generative AI** (Gemini, called only once per report to stay within the
free tier).

---

## Tech stack

| Layer | Tools |
|-------|-------|
| UI / App | Streamlit |
| Machine Learning | scikit-learn (TF-IDF + Logistic Regression) |
| Embeddings | sentence-transformers (multilingual MiniLM) |
| Vector search | NumPy in-memory cosine similarity |
| Agent workflow | LangGraph (7-node self-critique loop) |
| Generative AI | Google Gemini (via google-genai) |
| Charts | Plotly |

---

## Project structure

```
vibe-market-pulse/
├── app.py                  # Main Streamlit app (UI + ML pipeline)
├── requirements.txt        # Python dependencies
├── .python-version         # Pins Python 3.11 for stable deploys
├── .streamlit/
│   └── config.toml         # Theme + 100 MB upload limit
├── dataset/                # Sample datasets (for trying the app)
│   ├── ecommercereviews.csv            # base training data (English)
│   ├── adidasvsnike.csv
│   ├── tokopedia-product-reviews-2019.csv
│   └── datashopee.csv
└── src/
    └── ai/
        ├── vector_store.py     # In-memory NumPy semantic search (active)
        ├── aspect_clusters.py  # KMeans clustering of review themes
        ├── agent_graph.py      # LangGraph Agentic RAG workflow
        └── pgvector_store.py   # Persistent Postgres store (future, not wired)
```

---

## Running locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Add your Google Gemini API key (get one free at
   [Google AI Studio](https://aistudio.google.com/app/apikey)):
   ```toml
   # .streamlit/secrets.toml   (this file is git-ignored — never commit it)
   GEMINI_API_KEY = "your-key-here"
   ```

3. Start the app:
   ```bash
   streamlit run app.py
   ```

The sentiment prediction, charts, and Smart Search work **without** an API
key. Only the AI Consultant report needs the Gemini key.

---

## Using your own data

Upload any CSV that has a free-text review/comment column. The app tries hard
to read it correctly:

- Handles different encodings (UTF-8, Windows/Excel cp1252, Latin-1).
- Auto-detects the delimiter (comma, semicolon, tab, pipe).
- Skips a few malformed rows instead of failing.
- Auto-detects the text column, or lets you pick it manually.

If your file isn't about product reviews, the analysis simply won't be
meaningful — the app is built for market/customer-review analysis.

---

## Notes & limitations

- Data lives **in memory** for the current session — refreshing the page
  clears it. A persistent database (Postgres + pgvector) is scaffolded in
  `src/ai/pgvector_store.py` for a future version, but is not active yet.
- Gemini is called once per generated report to respect free-tier limits.
- Upload size is capped at 100 MB.

---

## Roadmap (ideas)

- Persistent storage with Postgres + pgvector (module already scaffolded).
- Optional user login.
- CI pipeline (lint + tests + security scan).
- Confidence calibration and model comparison.
