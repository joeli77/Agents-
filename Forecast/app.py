import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📈 Revenue Forecasting Agent", page_icon="📊", layout="wide")

st.title("🔮 AI Forecasting Agent with Prophet")

# File Upload
uploaded_file = st.file_uploader("📁 Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Data validation
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("❌ The Excel file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    # Prepare data
    df = df[['Date', 'Revenue']].dropna()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])

    st.subheader("📊 Uploaded Data Preview")
    st.dataframe(df.tail())

    # Prophet Forecasting
    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    st.subheader("📈 Forecasted Revenue")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # AI Commentary with Groq
    st.subheader("🤖 AI-Generated Forecast Commentary")

    data_for_ai = df.tail(24).to_json(orient='records')  # last 24 months for context
    prompt = f"""
    You are a top-tier Financial Analyst at a SaaS company. 
    Based on the following historical revenue data in JSON format, use trend analysis and your FP&A expertise to:
    - Summarize key historical trends in revenue.
    - Highlight potential reasons for the trends.
    - Suggest actionable business insights.
    Here is the data: {data_for_ai}
    """

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a senior financial analyst expert in revenue forecasting."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    ai_commentary = response.choices[0].message.content
    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    st.write(ai_commentary)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👆 Upload an Excel file to begin forecasting.")

