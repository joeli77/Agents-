# forecasting_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from prophet import Prophet
from groq import Groq
from dotenv import load_dotenv

# 🌍 Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="📈 Forecasting Agent with Prophet", page_icon="📊", layout="wide")
st.title("📈 Revenue Forecasting using Prophet")

# 📂 File Upload
uploaded_file = st.file_uploader("Upload your Excel file with 'Date' and 'Revenue' columns", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # 🔍 Basic Validation
    if 'Date' not in df.columns or 'Revenue' not in df.columns:
        st.error("The Excel file must contain 'Date' and 'Revenue' columns.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    st.subheader("📊 Uploaded Data Preview")
    st.write(df.head())

    # 🧠 Prophet Model
    prophet_df = df.rename(columns={"Date": "ds", "Revenue": "y"})
    model = Prophet()
    model.fit(prophet_df)

    # 📅 Forecast Horizon
    periods = st.slider("Select forecast horizon (months)", 1, 24, 6)
    future = model.make_future_dataframe(periods=periods * 30)  # approx months
    forecast = model.predict(future)

    st.subheader("📈 Forecasted Revenue")
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.subheader("🔍 Forecast Components")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # 🧠 Generate AI Commentary
    st.subheader("🤖 AI-Generated Forecast Commentary")

    # Only send last few rows for performance
    recent_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12).to_dict(orient="records")
    data_for_ai = pd.DataFrame(recent_data).to_json(orient="records")

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""
    You are the Head of FP&A at a SaaS company. Your task is to analyze the revenue forecast output below and provide:
    - Key insights about the trend.
    - Any seasonality or anomalies observed.
    - A summary fit for a CFO with clear business language.
    - Suggestions on revenue optimization.

    Here is the forecast data:
    {data_for_ai}
    """

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-8b-8192",
    )

    ai_commentary = response.choices[0].message.content
    st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
    st.write(ai_commentary)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("📁 Please upload an Excel file to begin forecasting.")
