from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from .data import load, persist, ROOT

def train():
    df = load()
    persist(df)
    X = df[["product","state","issue","response_days"]]
    y = df["delayed_response"]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["product","state","issue"]),
        ("num", StandardScaler(), ["response_days"]),
    ])
    rf = Pipeline([("pre", pre), ("model", RandomForestClassifier(
        n_estimators=180, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    ))])
    lr = Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    rf.fit(Xtr, ytr); lr.fit(Xtr, ytr)
    rf_auc = roc_auc_score(yte, rf.predict_proba(Xte)[:,1])
    lr_auc = roc_auc_score(yte, lr.predict_proba(Xte)[:,1])

    model_dir = ROOT/"models"
    report_dir = ROOT/"reports"
    model_dir.mkdir(exist_ok=True); report_dir.mkdir(exist_ok=True)
    joblib.dump(rf, model_dir/"delayed_response_model.joblib")
    metrics = {"random_forest_roc_auc": round(float(rf_auc), 4),
               "logistic_regression_roc_auc": round(float(lr_auc), 4),
               "rows": int(len(df))}
    (report_dir/"model_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(metrics)
