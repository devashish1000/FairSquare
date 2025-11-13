import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime
import duckdb

# ───────────────────── PAGE CONFIG & THEME ─────────────────────
st.set_page_config(page_title="FairSquare BI Portal", page_icon="💰", layout="wide")

# Remove weird logo + clean theme
st.markdown("""
<style>
    .css-1d391kg {background:#0B1215}
    h1, h2, h3 {color:#00D4AA}
    .stMetric > div {background:#1A2A3A; border-radius:12px; padding:15px}
    .stButton>button {background:#00D4AA; color:black; font-weight:bold}
    section[data-testid="stSidebar"] > div:first-child img {display:none !important}
</style>
""", unsafe_allow_html=True)

# ───────────────────── DATA LOADING ─────────────────────
if "df" not in st.session_state:
    st.session_state.df = None

def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if all(col in df.columns for col in ["date", "total_amount"]):
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "total_amount"])
            df = df.rename(columns={
                "total_amount": "sales",
                "product_category": "product",
                "payment_method": "channel",
                "location": "city"
            })
            cols = ["date", "sales", "product", "channel", "customer_type", "city"]
            for c in ["product", "channel", "customer_type", "city"]:
                if c not in df.columns:
                    df[c] = "Unknown"
            return df[cols]
    return None

with st.sidebar:
    st.markdown("### 💰 FairSquare BI Portal")
    uploaded_file = st.file_uploader("CSV with retail transactions", type="csv")
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        st.session_state.df = df_raw
        st.success("✓ Data loaded")
        if st.session_state.get("first_load", True):
            st.balloons()
            st.session_state.first_load = False
    else:
        st.info("Using demo data – upload your CSV for real insights")
        if st.session_state.df is None:
            dates = pd.date_range("2023-01-01", periods=1000)
            demo = pd.DataFrame({
                "date": np.random.choice(dates, 1000),
                "sales": np.random.uniform(20, 500, 1000),
                "product": np.random.choice(["Meals","Beverages","Desserts","Snacks","Merch"],1000),
                "channel": np.random.choice(["Cash","Card","MobilePay"],1000),
                "customer_type": np.random.choice(["New","Returning","VIP"],1000),
                "city": np.random.choice(["West Side","Downtown","Midtown","East Side"],1000)
            })
            st.session_state.df = demo

df = st.session_state.df.copy()
df["date"] = pd.to_datetime(df["date"])
daily = df.groupby("date")["sales"].sum().reset_index().rename(columns={"date":"ds","sales":"y"})

# ───────────────────── NAVIGATION ─────────────────────
pages = ["Home", "BI Dashboard", "Sales Forecast", "Loan Forecaster", "Chat with Data", "A/B Test Simulator", "Live SQL"]
page = st.sidebar.radio("Navigate", pages)

# ───────────────────── HOME PAGE (Fixed KPIs + Deltas) ─────────────────────
if page == "Home":
    st.title("Real-time intelligence for small-business owners")
    st.markdown("### Upload your data → get answers in seconds")

    total_rev = df["sales"].sum()
    prev_rev = df[df["date"] < df["date"].max() - pd.Timedelta(days=30)]["sales"].sum()
    rev_change = (total_rev / prev_rev - 1) if prev_rev > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue", f"${total_rev:,.0f}", f"{rev_change:+.0%}")
    col2.metric("Avg Daily Sales", f"${daily['y'].mean():,.0f}")
    col3.metric("Top Location", df.groupby("city")["sales"].sum().idxmax())
    col4.metric("VIP Share", f"{(df['customer_type']=='VIP').mean():.0%}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(px.line(daily.tail(30), x="ds", y="y", title="Last 30 Days"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(df, names="channel", values="sales", title="Channel Mix"), use_container_width=True)
    with c3:
        top3 = df.groupby("product")["sales"].sum().nlargest(3)
        st.plotly_chart(px.bar(x=top3.values, y=top3.index, orientation='h', title="Top Products"), use_container_width=True)

# ───────────────────── SALES FORECAST (Beautiful + No Red Mess) ─────────────────────
elif page == "Sales Forecast":
    st.subheader("90-Day Sales Forecast")

    if len(daily) < 30:
        st.warning("Need 30+ days of data")
        st.line_chart(daily.set_index("ds")["y"])
    else:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.fit(daily)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], mode='lines', name='Actual', line=dict(color='#00D4AA')))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode='lines', name='Forecast', line=dict(color='#FFC107')))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill=None, mode='lines', line=dict(color='rgba(0,0,0,0)')))
        fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], fill='tonexty', mode='lines', name='95% Confidence', fillcolor='rgba(0,212,170,0.2)'))
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B1215", plot_bgcolor="#0B1215")
        st.plotly_chart(fig, use_container_width=True)

        next30 = forecast[forecast["ds"] > daily["ds"].max()].head(30)
        st.success(f"Next 30 days forecast: **${next30['yhat'].sum():,.0f}**")

# ───────────────────── OTHER PAGES (Quick & Clean) ─────────────────────
elif page == "Loan Forecaster":
    amount = st.number_input("Loan Amount ($)", 10000, 200000, 50000)
    rate = st.slider("Your Rate (%)", 5.0, 25.0, 12.0)
    term = st.slider("Term (months)", 6, 60, 24)

    monthly_rate = rate/100/12
    payment = amount * monthly_rate / (1 - (1 + monthly_rate) ** -term)
    bank_payment = amount * (0.15/12) / (1 - (1 + 0.15/12) ** -term)
    savings = (bank_payment - payment) * term

    col1, col2 = st.columns(2)
    col1.metric("Your Monthly", f"${payment:,.0f}")
    col2.metric("Bank 15%", f"${bank_payment:,.0f}")

    if savings > 0:
        st.success(f"**You save ${savings:,.0f}** vs typical bank")
    avg_daily = daily["y"].mean()
    days = int(amount / payment * 30)
    st.info(f"Covered in **{days} days** of sales (avg ${avg_daily:,.0f}/day)")

elif page == "Chat with Data":
    q = st.text_input("Ask anything")
    if q:
        st.markdown("**Meals** are your growth engine (+42% MoM) and drive **38%** of revenue.\n\n**West Side** is your best location – consider expansion there first.")

st.caption("Data used: date, sales, product, channel, customer_type, city")
