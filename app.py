import streamlit as st
import base64
import plotly.express as px
import plotly.graph_objects as go
import tempfile

def export_pdf_from_html(html_content, filename):
    """Generate a downloadable PDF using HTML → base64."""
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f"""
        <a href="data:application/octet-stream;base64,{b64}" 
           download="{filename}" 
           style="padding:12px 18px; background:#FFC107; color:black; 
                  border-radius:8px; font-weight:bold; text-decoration:none;">
           Download PDF
        </a>
    """
    st.markdown(href, unsafe_allow_html=True)


# ───────────────────── TOP-RIGHT BUTTONS (NO FPDF) ─────────────────────
if st.session_state.df is not None and uploaded is not None:
    st.markdown("<div style='position:fixed; top:12px; right:12px; z-index:999'>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,1])

    # EXEC DECK (HTML-PDF)
    with c1:
        if st.button("Executive Deck"):
            # Generate charts
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp1:
                fig = px.line(daily.tail(90), x="ds", y="y", title="90-Day Trend")
                fig.update_layout(template="plotly_dark")
                fig.write_image(tmp1.name)
                trend_img = tmp1.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp2:
                m = Prophet(yearly_seasonality=True, weekly_seasonality=True)
                m.fit(daily)
                future = m.make_future_dataframe(periods=90)
                fc = m.predict(future)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=daily["ds"], y=daily["y"], name="Actual"))
                fig2.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast"))
                fig2.write_image(tmp2.name)
                fc_img = tmp2.name

            html = f"""
            <h1>FairSquare Executive Summary</h1>
            <h2>Financial Overview</h2>
            <p>Total Revenue: ${df['sales'].sum():,.0f}</p>
            <p>Average Daily Sales: ${daily['y'].mean():,.0f}</p>
            <p>Top Product: {df.groupby('product')['sales'].sum().idxmax()}</p>
            <p>Top Location: {df.groupby('city')['sales'].sum().idxmax()}</p>

            <h2>90-Day Trend</h2>
            <img src="data:image/png;base64,{base64.b64encode(open(trend_img,'rb').read()).decode()}">

            <h2>90-Day Forecast</h2>
            <img src="data:image/png;base64,{base64.b64encode(open(fc_img,'rb').read()).decode()}">
            """

            export_pdf_from_html(html, "FairSquare_Executive_Deck.html")  # user downloads HTML-as-PDF

    # WEEKLY SUMMARY
    with c2:
        if st.button("Weekly Summary"):
            last_week = daily.set_index("ds").resample("W-SUN").sum().iloc[-1]

            html = f"""
            <h1>Weekly Sales Report</h1>
            <h2>Week Ending {last_week.name.strftime('%b %d')}</h2>
            <p>Revenue: ${last_week['y']:,.0f}</p>
            <p>Key Actions:</p>
            <ul>
                <li>Run Meal Promo</li>
                <li>Re-engage VIP Customers</li>
                <li>Push MobilePay Adoption</li>
            </ul>
            """

            export_pdf_from_html(html, "FairSquare_Weekly_Report.html")

    st.markdown("</div>", unsafe_allow_html=True)
