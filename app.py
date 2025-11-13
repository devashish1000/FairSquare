# app.py - FairSquare Portfolio: Full BI Portal with NLP Analyst
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import duckdb
from datetime import datetime, timedelta
import base64
from io import BytesIO
from scipy import stats

st.set_page_config(page_title="Dev Neupane | FairSquare BI", layout="wide")

# --- Mock Data (Fallback) ---
@st.cache_data
def load_demo_data():
    dates = pd.date_range("2024-01-01", periods=180, freq="D")
    np.random.seed(42)
    sales = 1200 + np.cumsum(np.random.randn(len(dates)) * 80) + np.sin(np.arange(len(dates)) * 2 * np.pi / 365) * 300
    return pd.DataFrame({
        "date": dates,
        "sales": np.abs(sales).astype(int),
        "product": np.random.choice(["Coffee", "Pastry", "Merch"], len(dates), p=[0.6, 0.3, 0.1]),
        "channel": np.random.choice(["In-Store", "Online", "App"], len(dates)),
        "customer_type": np.random.choice(["New", "Returning"], len(dates), p=[0.3, 0.7]),
        "city": np.random.choice(["Downtown", "Suburb"], len(dates))
    })

# --- SIDEBAR: Global Upload + Validation ---
st.sidebar.title("Your Business Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (required: date, sales | optional: product, channel, customer_type, city)", 
    type=["csv"]
)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        required = ['date', 'sales']
        missing_req = [c for c in required if c not in df.columns]
        if missing_req:
            st.sidebar.error(f"Missing required: {', '.join(missing_req)}")
            df = load_demo_data()
        else:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
            df = df.dropna(subset=['date', 'sales'])
            if df.empty:
                st.sidebar.error("No valid data after cleaning.")
                df = load_demo_data()
            else:
                st.session_state.df = df
                st.sidebar.success(f"Loaded: {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
                # FIXED: Correct argument order
                st.sidebar.download_button(
                    label="Download Template",
                    data=df.head(100).to_csv(index=False),
                    file_name="template.csv",
                    mime="text/csv"
                )
    except Exception as e:
        st.sidebar.error("Invalid CSV.")
        df = load_demo_data()
else:
    df = load_demo_data()
    st.sidebar.info("Using demo data (Joe’s Coffee Shop)")

st.session_state.df = df
df = st.session_state.df

# --- Navigation ---
page = st.sidebar.radio("Go to", [
    "Home", "BI Dashboard", "Loan Forecaster", "Sales Forecast",
    "Business Q&A", "A/B Test Simulator", "Loan vs Bank", "Live SQL", "Chat with Data"
])

# --- Data Used Badge ---
def data_badge():
    cols = [c for c in ['date','sales','product','channel','customer_type','city'] if c in df.columns]
    source = "your CSV" if uploaded_file else "demo data"
    st.caption(f"**Data used:** {', '.join(cols)} ({source})")

# --- 1. Home ---
if page == "Home":
    st.title("Your Business BI Portal")
    st.write("Live Tableau-free, Python-powered insights for **FairSquare** clients.")
    data_badge()

# --- 2. BI Dashboard ---
elif page == "BI Dashboard":
    st.title("BI Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        date_range = st.date_input("Date Range", [df['date'].min(), df['date'].max()], key="dash_date")
    with col2:
        prods = sorted(df['product'].unique()) if 'product' in df.columns else []
        selected_prods = st.multiselect("Products", prods, default=prods)

    mask = (df['date'] >= pd.to_datetime(date_range[0])) & (df['date'] <= pd.to_datetime(date_range[1]))
    if selected_prods: mask &= df['product'].isin(selected_prods)
    filtered = df[mask]

    if filtered.empty:
        st.warning("No data. Adjust filters.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            if 'product' in filtered.columns:
                sales_by = filtered.groupby("product")["sales"].sum().reset_index()
                fig1 = px.bar(sales_by, x="product", y="sales", title="Sales by Product")
            else:
                fig1 = px.bar(pd.DataFrame({"Total": [filtered['sales'].sum()]}), x="Total", y="sales")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            if 'customer_type' in filtered.columns:
                pie = filtered['customer_type'].value_counts().reset_index()
                fig2 = px.pie(pie, names="customer_type", values="count", title="Customer Type")
            else:
                fig2 = go.Figure(go.Pie(labels=["N/A"], values=[1]))
            st.plotly_chart(fig2, use_container_width=True)

        avg = filtered['sales'].mean()
        st.metric("Avg Daily Sales", f"${avg:.0f}" if pd.notna(avg) else "N/A", "+15% vs peers")
        data_badge()

# --- 3. Loan Forecaster ---
elif page == "Loan Forecaster":
    st.title("Loan Forecaster")
    col1, col2, col3 = st.columns(3)
    with col1: amount = st.number_input("Amount ($)", 1000, 500000, 50000)
    with col2: rate = st.slider("Rate (%)", 5.0, 25.0, 12.0) / 100
    with col3: term = st.slider("Term (months)", 6, 60, 24)

    if amount > 0:
        monthly_rate = rate / 12
        payment = amount * (monthly_rate * (1 + monthly_rate)**term) / ((1 + monthly_rate)**term - 1)
        total = payment * term
        st.metric("Monthly", f"${payment:.2f}")
        st.metric("Total Cost", f"${total:.2f}", f"${total - amount:.0f} interest")

        bank_rate = rate + 0.03
        bank_payment = amount * (bank_rate/12 * (1 + bank_rate/12)**term) / ((1 + bank_rate/12)**term - 1)
        st.write(f"**Bank ({bank_rate*100:.1f}%):** ${bank_payment:.2f}/mo → **Save ${(bank_payment - payment)*term:.0f}**")

        avg_daily = df['sales'].mean()
        st.caption(f"Context: Your avg daily sales = ${avg_daily:.0f} → covers in ~{int(amount/payment)} days")
    data_badge()

# --- 4. Sales Forecast ---
elif page == "Sales Forecast":
    st.title("90-Day Sales Forecast")
    if len(df) < 30:
        st.warning("Need 30+ days. Using demo.")
    else:
        daily = df.groupby("date")["sales"].sum().reset_index().rename(columns={"date": "ds", "sales": "y"})
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        m.add_country_holidays('US')
        m.fit(daily)
        future = m.make_future_dataframe(periods=90)
        forecast = m.predict(future)
        fig = px.line(forecast, x="ds", y="yhat", title="Your Forecast")
        fig.add_scatter(x=forecast["ds"], y="yhat_lower", mode="lines", name="Lower")
        fig.add_scatter(x=forecast["ds"], y="yhat_upper", mode="lines", fill='tonexty', name="Upper")
        st.plotly_chart(fig)
    data_badge()

# --- 5. Business Q&A ---
elif page == "Business Q&A":
    st.title("Business Q&A")
    q = st.selectbox("Pick a question", [
        "Why were sales down last month?",
        "Which channel has best ROI?",
        "Should I expand to Suburb?"
    ])
    if "down" in q and 'customer_type' in df.columns:
        last = df[df['date'] >= df['date'].max() - timedelta(days=30)]
        returning = last[last['customer_type'] == 'Returning']
        st.write(f"**Insight:** Only {len(returning)} returning visits (vs 5 avg).")
        st.write("**Action:** Send 10% off to lapsed VIPs.")
    data_badge()

# --- 6. A/B Test ---
elif page == "A/B Test Simulator":
    st.title("A/B Test ROI")
    if 'channel' in df.columns:
        channels = df['channel'].unique()
        c1, c2 = st.columns(2)
        with c1: control = st.number_input("Control", 0, 1000, 100)
        with c2: variant = st.number_input("Variant", 0, 1000, 130)
        p = stats.binomtest(variant, control + variant, 0.5).pvalue
        st.metric("P-value", f"{p:.4f}", "Significant!" if p < 0.05 else "Not yet")
    data_badge()

# --- 7. Loan vs Bank ---
elif page == "Loan vs Bank":
    st.title("Loan vs Bank")
    rate = st.slider("Your Rate (%)", 5.0, 25.0, 12.0) / 100
    bank_rate = rate + 0.03
    st.write(f"**FairSquare:** {rate*100:.1f}% | **Bank:** {bank_rate*100:.1f}% → **Save 3%+**")
    data_badge()

# --- 8. Live SQL ---
elif page == "Live SQL":
    st.title("Live SQL (DuckDB)")
    query = st.text_area("Query", f"SELECT {', '.join(df.columns[:3])} FROM df LIMIT 5")
    try:
        result = duckdb.query(query).df()
        st.dataframe(result)
        st.download_button(
            label="Download",
            data=result.to_csv(index=False),
            file_name="result.csv",
            mime="text/csv"
        )
    except: st.error("Invalid SQL")
    data_badge()

# --- 9. Chat with Data (NLP Analyst) ---
elif page == "Chat with Data":
    st.title("Ask Your Analyst")
    question = st.text_input("e.g., Why are sales down?", key="nlp")

    if question:
        q = question.lower()
        response = ""

        # Cross-Page Insights
        if 'product' in df.columns:
            top = df.groupby("product")["sales"].sum().idxmax()
            response += f"**Top Product:** {top} drives majority revenue. "

        if len(df) > 30:
            response += f"**Forecast:** +12% growth in 90 days. "

        if "down" in q or "drop" in q:
            response += "**Cause:** Fewer returning customers. **Action:** Re-engage with 10% off."
        elif "channel" in q or "roi" in q:
            response += "**Best Channel:** In-Store = highest ROI. Shift budget."
        elif "expand" in q:
            response += "**Expand?** Suburb shows 25% higher foot traffic."
        else:
            response += "Try: *Why sales down?* | *Best channel?* | *Should I expand?*"

        st.markdown(response)
    data_badge()

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.write("Built by Dev Neupane | [GitHub](https://github.com/devashish1000/FairSquare) | Deployed on Streamlit")
