import os, sqlite3, json
import pandas as pd
import plotly.express as px
import streamlit as st
import joblib
from src.data import DB, load, persist
from src.llm import summarize_monthly

st.set_page_config(page_title="Complaint Resolution Analytics", layout="wide")
st.title("Predictive Complaint Resolution-Time Analytics")

@st.cache_data
def query(sql, params=()):
    con = sqlite3.connect(DB)
    try:
        return pd.read_sql_query(sql, con, params=params)
    finally:
        con.close()

if not DB.exists():
    df0 = load()
    persist(df0)

products = query("SELECT DISTINCT product FROM complaints ORDER BY product")["product"].tolist()
states = query("SELECT DISTINCT state FROM complaints ORDER BY state")["state"].tolist()
issues = query("SELECT DISTINCT issue FROM complaints ORDER BY issue")["issue"].tolist()

c1,c2,c3 = st.columns(3)
product = c1.multiselect("Product", products)
state = c2.multiselect("State", states)
issue = c3.multiselect("Issue", issues)

where, params = [], []
for col, values in [("product",product),("state",state),("issue",issue)]:
    if values:
        where.append(col + " IN (" + ",".join(["?"]*len(values)) + ")")
        params.extend(values)
clause = " WHERE " + " AND ".join(where) if where else ""

df = query("SELECT * FROM complaints" + clause, params)

a,b,c = st.columns(3)
a.metric("Complaints", f"{len(df):,}")
b.metric("Avg response days", f"{df.response_days.mean():.1f}")
c.metric("Delayed rate", f"{df.delayed_response.mean()*100:.1f}%")

left,right = st.columns(2)
with left:
    trend = df.groupby("product", as_index=False).size().rename(columns={"size":"complaints"})
    st.plotly_chart(px.bar(trend, x="product", y="complaints", title="Complaints by Product"), use_container_width=True)
with right:
    response = df.groupby("product", as_index=False).response_days.mean()
    st.plotly_chart(px.bar(response, x="product", y="response_days", title="Average Response Time"), use_container_width=True)

st.subheader("Monthly complaint trends")
df["month"] = pd.to_datetime(df["date_received"], errors="coerce").dt.to_period("M").astype(str)
monthly = df.groupby("month").agg(complaints=("complaint_id","count"), avg_response_days=("response_days","mean")).reset_index()
st.plotly_chart(px.line(monthly, x="month", y="complaints", markers=True), use_container_width=True)

if os.path.exists("models/delayed_response_model.joblib"):
    model = joblib.load("models/delayed_response_model.joblib")
    sample = df[["product","state","issue","response_days"]].head(1)
    if not sample.empty:
        st.subheader("Example delayed-response prediction")
        st.write(f"Predicted probability: {model.predict_proba(sample)[:,1][0]:.1%}")

if st.button("Generate monthly AI summary"):
    prompt = "Monthly metrics:\n" + monthly.tail(6).to_string(index=False)
    st.info(summarize_monthly(prompt))
