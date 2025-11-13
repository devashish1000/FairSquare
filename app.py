import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta
import duckdb
import base64
from fpdf import FPDF   # ✅ fpdf2 (safe)
import tempfile
import os

# ───────────────────── CONFIG & THEME ─────────────────────
st.set_page_config(page_title="FairSquare BI Portal", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .css-1d391kg {background:#0B1215 !important}
    h1,h2,h3 {color:#00D4AA; font-weight:600}
    .stMetric > div {background:#1A2A3A; border-radius:12px; padding:18px; border-left:6px solid #00D4AA}
    .stButton>button {background:#00D4AA; color:black; font-weight:bold; border-radius:12px}
    section[data-testid="stSidebar"] img {display:none !important}
    .pulse {animation: pulse 2s infinite}
    @keyframes pulse {0%{box-shadow:0 0 0 0 #00D4AA} 70%{box-shadow:0 0 0 20px transparent} 100%{box-shadow:0 0 0 0 transparent}}
    .report-button {background:#FFC107 !important; color:black !important}
</style>
""", unsafe_allow_html=True)

# ───────────────────── STATE & DATA ─────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
    st.session_state.first_load = True

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

with st.sidebar:
    st.markdown("### 💰 FairSquare BI Portal")
    uploaded = st.file_uploader("Upload retail transactions CSV", type="csv")
    df_raw = load_data(uploaded)
    
    if df_raw is not None:
        st.session_state.df = df_raw
        st.success("Real data loaded")
        if st.session_state.first_load:
            st.balloons()
            st.session_state.first_load = False
    else:
        st.info("Demo mode – upload your CSV")

# Load (or simulate) data
if st.session_state.df is None:
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
    df = st.session_state.df.copy()

df["date"] = pd.to_datetime(df["date"])
daily = df.groupby("date")["sales"].sum().reset_index().rename(columns={"date":"ds","sales":"y"})

# ───────────────────── EXECUTIVE PDF GENERATION FIX ─────────────────────
def pdf_bytes(pdf_obj):
    """fpdf2 requires encoding when outputting as string"""
    return pdf_obj.output(dest="S").encode("latin-1")


# ───────────────────── TOP-RIGHT BUTTONS ─────────────────────
if st.session_state.df is not None and uploaded is not None:
    with st.container():
        st.markdown("<div style='position:fixed; top:12px; right:12px; z-index:999'>", unsafe_allow_html=True)
        c1, c2 = st.columns([1,1])

        # EXEC DECK
        with c1:
            if st.button("Executive Deck"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 24)
                pdf.cell(0, 20, "FairSquare Executive Summary", ln=1, align="C")
                pdf.set_font("Helvetica", size=14)
                pdf.ln(10)
                pdf.cell(0, 12, f"Total Revenue: ${df['sales'].sum():,.0f}", ln=1)
                pdf.cell(0, 12, f"Avg Daily Sales: ${daily['y'].mean():,.0f}", ln=1)
                pdf.cell(0, 12, f"Top Product: {df.groupby('product')['sales'].sum().idxmax()}", ln=1)
                pdf.cell(0, 12, f"Top Location: {df.groupby('city')['sales'].sum().idxmax()}", ln=1)

                # Revenue trend
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 20)
                pdf.cell(0, 15, "Revenue Performance", ln=1)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    fig = px.line(daily.tail(90), x="ds", y="y", title="90-Day Trend")
                    fig.update_layout(template="plotly_dark")
                    fig.write_image(tmp.name)
                    pdf.image(tmp.name, w=180)

                # Forecast
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 20)
                pdf.cell(0, 15, "90-Day Forecast", ln=1)
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                m.fit(daily)
                future = m.make_future_dataframe(periods=90)
                fc = m.predict(future)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual", line=dict(color="#00D4AA")))
                    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast", line=dict(color="#FFC107")))
                    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"]))
                    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], fill="tonexty",
                                             fillcolor="rgba(0,212,170,0.15)", name="95% Confidence"))
                    fig.update_layout(template="plotly_dark", paper_bgcolor="#0B1215")
                    fig.write_image(tmp.name)
                    pdf.image(tmp.name, w=180)

                st.download_button(
                    "Download Executive Deck",
                    data=pdf_bytes(pdf),
                    file_name="FairSquare_Executive_Deck.pdf",
                    mime="application/pdf"
                )

        # WEEKLY SUMMARY
        with c2:
            if st.button("Weekly Summary"):
                last_week = daily.set_index("ds").resample("W-SUN").sum().iloc[-1]
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 24)
                pdf.cell(0, 20, f"Week Ending {last_week.name.strftime('%b %d')}", ln=1, align="C")
                pdf.set_font("Helvetica", size=16)
                pdf.ln(15)
                pdf.cell(0, 12, f"Revenue: ${last_week['y']:,.0f}", ln=1)
                pdf.cell(0, 12, "Key Actions: Run Meal promo • Re-engage VIPs • Push MobilePay", ln=1)

                st.download_button(
                    "Download Weekly Report",
                    data=pdf_bytes(pdf),
                    file_name=f"FairSquare_Weekly_{last_week.name.strftime('%Y_W%W')}.pdf",
                    mime="application/pdf"
                )

        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── SETTINGS ─────────────────────
with st.container():
    st.markdown("<div style='position:fixed; bottom:10px; left:10px; z-index:999'>", unsafe_allow_html=True)
    with st.expander("Settings"):
        st.selectbox("Theme", ["Dark","Light"])
        st.selectbox("Currency", ["USD","EUR","GBP"])
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── HOME PAGE ─────────────────────
st.title("Real-time intelligence for small-business owners")
st.markdown("### Upload your data → get answers in seconds")

if uploaded is None:
    st.markdown("<div class='pulse' style='text-align:center'><h2 style='color:#00D4AA'>↑ Upload CSV to Activate Full Portal</h2></div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${df['sales'].sum():,.0f}")
c2.metric("Avg Daily Sales", f"${daily['y'].mean():,.0f}")
c3.metric("Top Location", df.groupby("city")["sales"].sum().idxmax())
c4.metric("VIP Share", f"{(df['customer_type']=='VIP').mean():.0%}")

st.caption("Data: date, sales, product, channel, customer_type, city • Built by Dev Neupane")
