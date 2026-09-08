from pathlib import Path
import random
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB = DATA / "complaints.db"

def synthetic(n=12000):
    random.seed(42)
    products = ["Credit card","Mortgage","Checking account","Debt collection","Student loan"]
    states = ["CA","NY","TX","FL","IL","WA","NJ","PA"]
    issues = ["Billing dispute","Fees","Payment processing","Account access","Loan servicing"]
    rows = []
    start = pd.Timestamp("2024-01-01")
    for i in range(n):
        date = start + pd.Timedelta(days=random.randint(0, 730))
        days = max(1, int(random.gauss(10, 7)))
        rows.append({
            "complaint_id": i + 1,
            "date_received": date.date().isoformat(),
            "product": random.choice(products),
            "state": random.choice(states),
            "issue": random.choice(issues),
            "response_days": days,
            "timely_response": "No" if days > 15 else "Yes"
        })
    return pd.DataFrame(rows)

def load():
    path = DATA / "consumer_complaints.csv"
    if path.exists():
        df = pd.read_csv(path)
        return normalize(df)
    return synthetic()

def normalize(df):
    df = df.copy()
    rename = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc == "date received": rename[c] = "date_received"
        elif lc == "product": rename[c] = "product"
        elif lc == "state": rename[c] = "state"
        elif lc == "issue": rename[c] = "issue"
        elif lc == "complaint id": rename[c] = "complaint_id"
        elif lc == "timely response": rename[c] = "timely_response"
    df = df.rename(columns=rename)
    if "response_days" not in df:
        # CFPB files do not always contain an exact duration field.
        # Derive a defensible proxy when date columns are available.
        recv = pd.to_datetime(df.get("date_received"), errors="coerce")
        sent = pd.to_datetime(df.get("date_sent_to_company"), errors="coerce")
        df["response_days"] = (sent - recv).dt.days
    for c in ["product","state","issue"]:
        if c not in df: df[c] = "Unknown"
        df[c] = df[c].fillna("Unknown").astype(str)
    df["response_days"] = pd.to_numeric(df["response_days"], errors="coerce").fillna(0).clip(lower=0)
    df["delayed_response"] = (df["response_days"] > 15).astype(int)
    return df

def persist(df):
    DATA.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    df.to_sql("complaints", con, if_exists="replace", index=False)
    con.execute("CREATE INDEX IF NOT EXISTS idx_product ON complaints(product)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_state ON complaints(state)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_issue ON complaints(issue)")
    con.commit()
    con.close()
