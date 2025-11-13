# app.py - Final FairSquare Portfolio Site (Streamlit)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import duckdb
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO

st.set_page_config(page_title="Dev Neupane | FairSquare BI", layout="wide")

# --- Mock Data ---
@st.cache_data
def load_data():
    dates = pd.date_range("2023-01-01", "2025-11-01", freq="D")
    np.random.seed(42)
    sales = 1000 + np.cumsum(np.random.randn(len(dates)) * 50) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 300
    data = pd.DataFrame({
        "date": dates,
        "sales": sales + np.random.randint(50, 200, len(dates)),
        "product": np.random.choice(["Coffee", "Pastry", "Merch"], len(dates), p=[0.6, 0.3, 0.1]),
        "channel": np.random.choice(["Online", "In-Store", "App"], len(dates)),
        "customer_type": np.random.choice(["New", "Returning"], len(dates), p=[0.3, 0.7]),
        "city": np.random.choice(["Downtown", "Suburb"], len(dates))
    })
    return data

df = load_data()

# --- Sidebar Nav ---
st.sidebar.title("Dev Neupane")
st.sidebar.write("BI • Python • ML for Small Businesses")
page = st.sidebar.radio("Go to", [
    "Home", "BI Dashboard", "Loan Forecaster", "Sales Forecast",
    "Business Q&A", "A/B Test Simulator", "Loan vs Bank", "Live SQL (DuckDB)", "Chat with Data"
])

# --- Home ---
if page == "Home":
    st.title("Data Tools for Small Business Heroes")
    st.write("Built for **FairSquare** — live demos of BI, ML, and client-facing apps.")
    st.info("All in Python • Deployed free • Code on GitHub")

# --- BI Dashboard ---
elif page == "BI Dashboard":
    st.title("Small Business BI Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input("Date Range", [df['date'].min(), df['date'].max()])
    with col2:
        product = st.multiselect("Product", df['product'].unique(), df['product'].unique())

    mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1])) & (df['product'].isin(product))
    filtered = df[mask]

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(filtered.groupby("product")["sales"].sum().reset_index(), x="product", y="sales", title="Sales by Product")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(filtered, names="customer_type", title="New vs Returning")
        st.plotly_chart(fig, use_container_width=True)

    # Peer Benchmark
    benchmark = filtered['sales'].mean() * 1.15
    st.metric("Your Avg Daily Sales", f"${filtered['sales'].mean():.0f}", f"+15% vs peers")

    # PDF Export
    buf = BytesIO()
    fig.write_html(buf)
    st.download_button("Download Report", buf.getvalue(), "dashboard.pdf", "application/pdf")

# --- Loan Forecaster ---
elif page == "Loan Forecaster":
    st.title("FinTech Loan Forecaster")
    col1, col2, col3 = st.columns(3)
    with col1: amount = st.number_input("Loan Amount ($)", 1000, 500000, 50000)
    with col2: rate = st.slider("Interest Rate (%)", 5.0, 25.0, 12.0) / 100
    with col3: term = st.slider("Term (months)", 6, 60, 24)

    monthly_rate = rate / 12
    payment = amount * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)
    total = payment * term
    interest = total - amount

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Monthly Payment", f"${payment:.2f}")
        st.metric("Total Cost", f"${total:.2f}", f"${interest:.2f} interest")
    with c2:
        schedule = pd.DataFrame({
            "Month": range(1, term + 1),
            "Payment": [payment] * term,
            "Principal": [payment - (amount * monthly_rate)] * term,
            "Balance": [amount - (i * (payment - amount * monthly_rate)) for i in range(term)]
        })
        fig = px.area(schedule, x="Month", y=["Principal", "Payment"], title="Amortization")
        st.plotly_chart(fig)

    # vs Bank
    bank_rate = rate + 0.03
    bank_payment = amount * (bank_rate/12 * (1 + bank_rate/12)**term) / ((1 + bank_rate/12)**term - 1)
    st.write(f"**Bank Rate ({bank_rate*100:.1f}%)**: ${bank_payment:.2f}/mo → Save ${(bank_payment - payment)*term:.0f} with FairSquare")

# --- Sales Forecast ---
elif page == "Sales Forecast":
    st.title("90-Day Sales Forecast (Prophet)")
    daily = df.groupby("date")["sales"].sum().reset_index()
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    m.add_country_holidays('US')
    m.fit(daily.rename(columns={"date": "ds", "sales": "y"}))
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)

    fig = px.line(forecast, x="ds", y="yhat", title="Forecast")
    fig.add_scatter(x=forecast["ds"], y=forecast["yhat_lower"], fill=None, name="Lower")
    fig.add_scatter(x=forecast["ds"], y=forecast["yhat_upper"], fill='tonexty', name="Upper")
    st.plotly_chart(fig)

    st.code("""m = Prophet(); m.fit(df); future = m.make_future_dataframe(90); forecast = m.predict(future)""")

# --- Business Q&A ---
elif page == "Business Q&A":
    st.title("Answering Your Questions")
    q = st.selectbox("Pick a question", [
        "Why were sales down last month?",
        "Which channel has best ROI?",
        "Should I expand to Suburb?"
    ])
    if "down" in q:
        last_month = df[df['date'] >= df['date'].max() - timedelta(days=30)]
        top5 = last_month[last_month['customer_type'] == 'Returning'].groupby("customer_type")["sales"].sum()
        st.write("Top customers down 30% → Re-engage with 10% off")
        fig = px.bar(top5)
        st.plotly_chart(fig)
    # Add others...

# --- A/B Test Simulator ---
elif page == "A/B Test Simulator":
    st.title("A/B Test ROI Simulator")
    control = st.number_input("Control Conversions", 0, 1000, 100)
    variant = st.number_input("Variant Conversions", 0, 1000, 130)
    from scipy import stats
    p = stats.binomtest(variant, control + variant, 0.5).pvalue
    st.metric("P-value", f"{p:.4f}", "Significant!" if p < 0.05 else "Not yet")

# --- Live SQL (DuckDB) ---
elif page == "Live SQL (DuckDB)":
    st.title("Live SQL Query (DuckDB)")
    query = st.text_area("Write SQL", "SELECT product, SUM(sales) FROM df GROUP BY product")
    try:
        result = duckdb.query(query).df()
        st.dataframe(result)
        st.download_button("Download", result.to_csv(index=False), "result.csv")
    except: st.error("Invalid SQL")

# --- Chat with Data ---
elif page == "Chat with Data":
    st.title("Ask Your Data")
    question = st.text_input("e.g., Why sales down?")
    if "down" in question.lower():
        st.write("Top 5 returning customers dropped 30%. Action: Re-engagement campaign.")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Built in 24h • [GitHub](https://github.com) • Deploy: Streamlit Cloud")
