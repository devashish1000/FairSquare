import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import duckdb
import base64
import tempfile
import os
from pdfme import build_pdf

st.set_page_config(page_title="FairSquare BI Portal", page_icon="💰", layout="wide")

# ───────────────────── STYLES ─────────────────────
st.markdown("""
<style>
    .css-1d391kg {background:#0B1215 !important}
    h1,h2,h3 {color:#00D4AA; font-weight:600}
    .stMetric > div {background:#1A2A3A; border-radius:12px; padding:18px; border-left:6px solid #00D4AA}
    .stButton>button {background:#00D4AA; color:black; font-weight:bold; border-radius:12px}
    section[data-testid="stSidebar"] img {display:none !important}
    .pulse {animation: pulse 2s infinite}
    @keyframes pulse {0%{box-shadow:0 0 0 0 #00D4AA} 70%{box-shadow:0 0 0 20px transparent} 100%{box-shadow:0 0 0 0 transparent}}
</style>
""", unsafe_allow_html=True)


# ───────────────────── SESSION INIT ─────────────────────
if "df" not in st.session_state:
    st.session_state["df"] = None
    st.session_state["first_load"] = True


# ───────────────────── DATA LOADER ─────────────────────
def load_data(file):
    if file is not None:
        try:
            df = pd.read_csv(file)
            if {"date", "total_amount"}.issubset(df.columns):
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date", "total_amount"])
                df.rename(columns={
                    "total_amount": "sales",
                    "product_category": "product",
                    "payment_method": "channel",
                    "location": "city"
                }, inplace=True)
                for c in ["product","channel","customer_type","city"]:
                    if c not in df.columns: df[c] = "Unknown"
                return df[["date","sales","product","channel","customer_type","city"]]
        except:
            pass
    return None


# ───────────────────── SIDEBAR ─────────────────────
with st.sidebar:
    st.markdown("### 💰 FairSquare BI Portal")
    uploaded = st.file_uploader("Upload retail transactions CSV", type="csv")
    df_raw = load_data(uploaded)

    if df_raw is not None:
        st.session_state["df"] = df_raw
        st.success("Data loaded")
        if st.session_state["first_load"]:
            st.balloons()
            st.session_state["first_load"] = False
    else:
        st.info("Demo mode – upload your CSV")


# ───────────────────── LOAD OR SIMULATE DATA ─────────────────────
if st.session_state.get("df") is None:
    dates = pd.date_range("2023-01-01", periods=1200)
    df = pd.DataFrame({
        "date": np.random.choice(dates, 1200),
        "sales": np.random.uniform(25, 800, 1200),
        "product": np.random.choice(["Meals","Beverages","Desserts","Snacks","Merch"],1200),
        "channel": np.random.choice(["Cash","Card","MobilePay"],1200),
        "customer_type": np.random.choice(["New","Returning","VIP"],1200),
        "city": np.random.choice(["West Side","Downtown","Midtown","East Side"],1200)
    })
else:
    df = st.session_state.get("df").copy()

df["date"] = pd.to_datetime(df["date"])
daily = df.groupby("date")["sales"].sum().reset_index().rename(columns={"date":"ds","sales":"y"})


# ───────────────────── PDF BUILDER (pdfme) ─────────────────────
def create_pdf(sections, path):
    """
    sections = list of {"type": "text"/"image", "content": ...}
    pdfme consumes simple dict arrays.
    """
    doc = []

    for section in sections:
        if section["type"] == "text":
            doc.append({"text": section["content"], "size": 14})
            doc.append({"text": "\n"})
        elif section["type"] == "image":
            doc.append({"image": section["content"], "width": 500})
            doc.append({"text": "\n"})

    build_pdf({"sections": doc}, path)


# ───────────────────── TOP-RIGHT BUTTONS ─────────────────────
if st.session_state.get("df") is not None and uploaded is not None:
    st.markdown("<div style='position:fixed; top:12px; right:12px; z-index:999;'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])

    # EXECUTIVE DECK
    with c1:
        if st.button("Executive Deck"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp1:
                fig = px.line(daily.tail(90), x="ds", y="y", title="90-Day Trend")
                fig.update_layout(template="plotly_dark")
                fig.write_image(tmp1.name)
                img_trend = tmp1.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp2:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                m.fit(daily)
                future = m.make_future_dataframe(periods=90)
                fc = m.predict(future)
                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual"))
                fig_fc.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast"))
                fig_fc.write_image(tmp2.name)
                img_fc = tmp2.name

            pdf_path = "Executive_Deck.pdf"

            sections = [
                {"type": "text", "content": "FAIRSQUARE EXECUTIVE SUMMARY\n"},
                {"type": "text", "content": f"Total Revenue: ${df['sales'].sum():,.0f}"},
                {"type": "text", "content": f"Average Daily Sales: ${daily['y'].mean():,.0f}"},
                {"type": "text", "content": f"Top Product: {df.groupby('product')['sales'].sum().idxmax()}"},
                {"type": "text", "content": f"Top Location: {df.groupby('city')['sales'].sum().idxmax()}"},
                {"type": "image", "content": img
