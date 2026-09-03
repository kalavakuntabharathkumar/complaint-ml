# Predictive Complaint Resolution-Time Analytics Tool

A portfolio-ready analytics and ML application using CFPB Consumer Complaint data.
It trains a Random Forest classifier for delayed responses, exposes interactive Streamlit
analytics, and optionally generates plain-English trend summaries through the OpenAI API.

## Tech Stack
Python, Scikit-learn, Pandas, SQL, Streamlit, Plotly, OpenAI API, Joblib

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_training.py
streamlit run app.py
```

The project includes a reproducible synthetic fallback dataset, so the app can run without a
large Kaggle download. For the real CFPB dataset, place a CSV at `data/consumer_complaints.csv`
and run training again.

## Outputs
- `models/delayed_response_model.joblib`
- `data/complaints.db`
- `reports/model_metrics.json`

## Prediction Target
`delayed_response = 1` when the company response exceeds a configurable threshold in days.
The sample pipeline uses 15 days.

## App Features
- Product/state/issue filters
- Complaint volume and response-time analytics
- SQL-backed aggregations
- Cached queries
- Random Forest prediction
- Optional OpenAI monthly trend summary
- Retry + validation handling for LLM output
