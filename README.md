# ReviewPulse

ReviewPulse is a lightweight Streamlit app for monthly e-commerce review analysis. It provides an overview dashboard and a separate chat tab for evidence-based follow-up questions.

## What it does

- Reads CSV and Excel exports and lets the user map review, product, date, and rating columns.
- Shows review sentiment, product mentions, and common feedback themes.
- Answers follow-up questions in English, Hindi, or Hinglish and cites matching review rows.
- Offers two analysis modes:
  - **OpenAI API:** `gpt-4o-mini` classifies reviews in small batches and answers chat questions. Model inference runs remotely; no large model is downloaded into Streamlit Cloud.
  - **Local fallback:** a small transparent Hindi/Hinglish keyword lexicon, with no model download or API key. Treat it as approximate; it will miss nuance, sarcasm, and uncommon spellings.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload `sample_reviews.csv` to try the interface. Add an OpenAI API key in the sidebar for contextual Hindi/Hinglish classification and chat. The key is held in the Streamlit session and is not written to disk by this app.

## Streamlit Community Cloud

Deploy this repository with `app.py` as the entry point and `requirements.txt` as the dependency file. The app runs without a key using the approximate local fallback. To enable model-backed analysis, add `OPENAI_API_KEY` in Streamlit secrets or enter a key in the sidebar. The OpenAI API receives the review text in that mode and API usage may incur charges; do not upload sensitive personal information.

## Input expectations and limits

- CSV, XLSX, and XLS uploads are accepted. Recommended columns: review text, product, date, and rating.
- Up to 1,200 non-empty reviews are analyzed per upload to keep the hosted app responsive. Review text is capped at 1,200 characters per row.
- Ratings and dates are retained for exploration/display; the first version does not yet use ratings as sentiment labels or compare separate months.
- The local fallback is intentionally small and explainable, not a replacement for multilingual model analysis. For more reliable Hindi/Hinglish results, use API mode.
- No uploaded file is deliberately persisted by app code; Streamlit session memory holds the current analysis.

## Before using with business data

The app is an MVP, not a compliance-reviewed service. Confirm your data-sharing requirements before sending customer review text to an external model API. For deployment, keep API credentials in Streamlit secrets or the session input, never commit them.

