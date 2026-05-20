A Program For market analyst.
Upload your file(csv/xls). And got your insight and statictic information of your file

.
Overall Structure (top to bottom)
1. st.title()              — "🛍️ Hybrid E-Commerce Sentiment Analyzer"
2. st.caption()            — plain text subtitle
3. Sidebar                 — file uploader + Gemini status + tech stack text
4. render_model_performance() — 4 metric cards + expander
5. st.divider()
6. st.subheader()          — "🔍 Predict Sentiments on Your Data"
   ├── st.info() banners   — domain/language detection
   ├── st.success()        — domain training result
   ├── Sentiment chart     — pie chart (left) + bar chart (right) in columns
   ├── st.divider()
   ├── Product Intelligence section
   │   ├── Viral Products  — horizontal bar (green)
   │   └── Red Flag cats   — donut pie (reds)
   ├── st.divider()
   └── Interactive Explorer — text_input + radio + st.dataframe() + download button
7. st.divider()
8. st.selectbox()          — model selector
9. st.subheader()          — "🤖 Ask AI Consultant"
10. st.button()            — "→ Generate Insights"
11. AI report output       — st.markdown()
12. st.divider() + st.caption() — footer
