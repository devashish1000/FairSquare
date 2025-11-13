import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import duckdb
from datetime import datetime, timedelta
import base64
from fpdf import FPDF
import tempfile
import os

# ───────────────────── CONFIG & THEME ─────────────────────
st.set_page_config(page_title="FairSquare BI Portal", page_icon="💰", layout="wide")

st.markdown("""
<style>
    .css-1d391kg {background:#0B1215 !important}
    h1, h2, h3 {color:#00D4AA; font-weight:600}
    .stMetric > div {background:#1A2A3A; border-radius:12px; padding:18px; border-left:6px solid #00D4AA}
    .stButton>button {background:#00D4AA; color:black; font-weight:bold; border-radius:12px; height:3em}
    section[data-testid="stSidebar"] img {display:none !important}
    .pulse {animation: pulse 2s infinite}
    @keyframes pulse {0%{box-shadow:0 0 0 0 #00D4AA} 70%{box-shadow:0 0 0 20px transparent} 100%{box-shadow:0 0 0 0 transparent}}
    .report-button {background:#FFC107 !important; color:black !important}
</style>
""", unsafe_allow_html=True)

# ───────────────────── STATE & DATA LOADING ─────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
    stalletate.first_load = True

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
        except: pass
    return None

with st.sidebar:
    st.markdown("### 💰 FairSquare BI Portal")
    uploaded = st.file_uploader("Upload retail transactions CSV", type="csv")
    df_raw = load_data(uploaded)
    
    if df_raw is not None:
        st.session_state.df = df_raw
        st.success("✓ Real data loaded")
        if st.session_state.first_load:
            st.balloons()
            st.session_state.first_load = False
    else:
        st.info("Demo mode – upload your CSV for real insights")

# Use real data or fallback to demo
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
weekly = daily.set_index("ds").resample("W-SUN").sum().reset_index()

# ───────────────────── TOP-RIGHT: PRESENTATION + WEEKLY REPORT ─────────────────────
if st.session_state.df is not None and uploaded is not None:
    top_right = st.container()
    with top_right:
        st.markdown("<div style='position:fixed; top:12px; right:12px; z-index:999'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Executive Deck", type="primary"):
                # ───── 5-Slide McKinsey Deck ─────
                class PDF(FPDF):
                    def header(self): self.set_font('Helvetica','B',18); self.cell(0,15,"FairSquare Executive Summary", ln=1, align='C')
                    def footer(self): self.set_y(-15); self.set_font('Helvetica','I',9); self.cell(0,10,f'Page {self.page_no()} • Confidential',0,0,'C')
                pdf = PDF(); pdf.add_page(); pdf.set_font("Helvetica", size=12)
                # Slide 1
                pdf.set_font('Helvetica','B',22); pdf.cell(0,20,"Executive Summary", ln=1)
                pdf.set_font('Helvetica', size=14)
                pdf.ln(5); pdf.cell(0,10,f"Total Revenue: ${df['sales'].sum():,.0f}", ln=1)
                pdf.cell(0,10,f"Avg Daily Sales: ${daily['y'].mean():,.0f}", ln=1)
                pdf.cell(0,10,f"Top Product: {df.groupby('product')['sales'].sum().idxmax()}", ln=1)
                pdf.cell(0,10,f"Top Location: {df.groupby('city')['sales'].sum().idxmax()}", ln=1)
                # Slide 2–5 (charts)
                for title, fig in [
                    ("Revenue Trend", px.line(daily.tail(90), x="ds", y="y")),
                    ("Product Mix", px.pie(df, names="product", values="sales")),
                    ("Channel Mix", px.pie(df, names="channel", values="sales")),
                    ("90-Day Forecast", go.Figure().add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual")).add_trace(go.Scatter(x=Prophet().fit(daily).make_future_dataframe(90).predict().iloc[-90:], x="ds", y="yhat", name="Forecast")))
                ]:
                    pdf.add_page(); pdf.set_font('Helvetica','B',20); pdf.cell(0,15,title, ln=1)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B1215")
                        fig.write_image(tmp.name, width=1000, height=600)
                        pdf.image(tmp.name, x=20, w=170)
                b64 = base64.b64encode(pdf.output(dest="S")).decode()
                st.download_button("Download Executive Deck (PDF)", data=base64.b64decode(b64), file_name="FairSquare_Executive_Deck.pdf", mime="application/pdf")
        with col2:
            if st.button("Weekly Summary", type="secondary", help="Auto-generates Monday morning report"):
                # ───── Weekly Snapshot ─────
                last_week = weekly.iloc[-1]
                prev_week = weekly.iloc[-2]
                pdf = FPDF(); pdf.add_page()
                pdf.set_font('Helvetica','B',24); pdf.cell(0,20,f"Week Ending {last_week['ds'].strftime('%b %d')}", ln=1, align='C')
                pdf.set_font('Helvetica', size=16)
                pdf.ln(10)
                pdf.cell(0,12,f"Revenue: ${last_week['y']:,.0f}  ({(last_week['y']/prev_week['y']-1):+0.0%} vs last week)", ln=1)
                pdf.cell(0,12,f"Top Product: {df[df['date'].dt.isocalendar().week == last_week['ds'].weekofyear].groupby('product')['sales'].sum().idxmax()}", ln=1)
                pdf.cell(0,12,"Key Actions: Run Meal promo • Re-engage VIPs • Shift to MobilePay", ln=1)
                b64 = base64.b64encode(pdf.output(dest="S")).decode()
                st.download_button("Download Weekly Summary", data=base64.b64decode(b64), file_name=f"FairSquare_Weekly_{last_week['ds'].strftime('%Y_W%W')}.pdf", mime="application/pdf")
        st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── BOTTOM-LEFT SETTINGS ─────────────────────
with st.container():
    st.markdown("<div style='position:fixed; bottom:10px; left:10px; z-index:999'>", unsafe_allow_html=True)
    with st.expander("⚙️ Settings"):
        st.selectbox("Theme", ["Dark","Light"], key="theme")
        st.selectbox("Currency", ["USD","EUR","GBP"], key="currency")
        st.button("Clear cache", on_click=lambda: st.session_state.clear())
    st.markdown("</div>", unsafe_allow_html=True)

# ───────────────────── HOME PAGE ─────────────────────
st.title("Real-time intelligence for small-business owners")
st.markdown("### Upload your data → get answers in seconds")

if uploaded is None:
    st.markdown("<div class='pulse' style='text-align:center'><h2 style='color:#00D4AA'>↑ Upload CSV to Activate</h2></div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
total_rev = df["sales"].sum()
c1.metric("Total Revenue", f"${total_rev:,.0f}")
c2.metric("Avg Daily Sales", f"${daily['y'].mean():,.0f}")
c3.metric("Top Location", df.groupby("city")["sales"].sum().idxmax())
c4.metric("VIP Share", f"{(df['customer_type']=='VIP').mean():.0%}")

c1, c2, c3 = st.columns(3)
with c1:
    fig = px.line(daily.tail(30), x="ds", y="y", title="Last 30 Days")
    fig.update_traces(hovertemplate="Sales: $%{y:,.0f}")
    fig.update_xaxes(title="Date"); fig.update_yaxes(title="Daily Sales ($)", tickformat="$,"); fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

# ... (other pages: Sales Forecast, Loan Forecaster, Chat – all with same pro formatting)

st.caption("Data: date, sales, product, channel, customer_type, city • Built by Dev Neupane")
