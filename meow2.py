import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime

# --------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------
st.set_page_config(page_title="Portfolio Analyzer", page_icon="📈", layout="wide")

INITIAL_INVESTMENT = 50000   # Per stock (as per your project)

STOCKS = {
    "HCLTECH.NS":  "HCL Technologies",
    "ADANIENT.NS": "Adani Enterprises",
    "TECHM.NS":    "Tech Mahindra",
    "INFY.NS":     "Infosys",
    "WIPRO.NS":    "Wipro",
    "OFSS.NS":     "Oracle Financial Services",
    "MPHASIS.NS":  "Mphasis",
    "LTIM.NS":     "LTIMindtree",
    "PERSISTENT.NS": "Persistent Systems",
    "TCS.NS":      "Tata Consultancy Services"
}

DEFAULT_START = "2024-06-03"   # As used in your PDF report
DEFAULT_END = pd.Timestamp.today().strftime("%Y-%m-%d")


# --------------------------------------------------------------------
# FUNCTION — Robust NSE downloader
# --------------------------------------------------------------------
def fetch_data(symbols, start_date, end_date):
    data = pd.DataFrame()
    progress = st.progress(0)

    for i, symbol in enumerate(symbols):
        success = False

        for attempt in range(3):
            try:
                df = yf.download(
                    symbol, 
                    start=start_date, 
                    end=end_date,
                    progress=False,
                    threads=False,
                    timeout=30
                )

                if not df.empty:
                    data[symbol] = df["Open"].astype(float)
                    success = True
                    break

            except Exception as e:
                st.write(f"Attempt {attempt+1} failed for {symbol}: {e}")

        if not success:
            st.error(f"❌ Could not fetch: {symbol}")

        progress.progress((i + 1) / len(symbols))

    progress.empty()

    if data.empty:
        st.error("No stock data downloaded. Please check date range or try later.")
        return pd.DataFrame()

    return data.ffill().dropna()


# --------------------------------------------------------------------
# CALCULATION FUNCTIONS
# --------------------------------------------------------------------
def calculate_portfolio_value(initial_investment, data):
    weights = initial_investment / data.iloc[0]
    portfolio_value = (data * weights).sum(axis=1)

    df = data.copy()
    df["portfolio_value"] = portfolio_value
    return df, weights


def calculate_daily_returns(data):
    return data.pct_change().dropna()


def calculate_sharpe_ratio(series):
    if series.std() == 0:
        return 0
    return (series.mean()) / series.std()


def calculate_daily_sharpe_ratios(returns):
    ratios = pd.DataFrame(index=returns.index, columns=returns.columns)
    for i in range(1, len(returns)):
        window = returns.iloc[:i]
        ratios.iloc[i] = window.apply(calculate_sharpe_ratio, axis=0)
    return ratios


def plot_daily_sharpe_differences(returns):
    daily_sharpe = returns.apply(calculate_sharpe_ratio, axis=1)
    diff = daily_sharpe.diff()
    fig = px.line(x=diff.index, y=diff.values,
                  title="Daily Differences in Sharpe Ratios",
                  labels={"x": "Date", "y": "Sharpe Δ"})
    return fig


def optimize_portfolio(returns):
    n = len(returns.columns)
    mean_r = returns.mean()
    cov = returns.cov()

    def neg_sharpe(weights):
        ret = np.dot(weights, mean_r)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        return -(ret / vol) if vol != 0 else 999

    initial = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    result = minimize(neg_sharpe, initial, bounds=bounds, constraints=cons)

    if result.success:
        return result.x
    return initial


# --------------------------------------------------------------------
# PAGES
# --------------------------------------------------------------------
def home_page():
    st.title("📈 Portfolio Analyzer – Rebuilt Version")
    st.markdown("""
    This is a **fresh, fully optimized, and cloud-stable version** of your
    Portfolio Optimization App (based on your project report).

    ### Features
    - Robust NSE data fetching  
    - Portfolio Optimization (Mean-Variance)  
    - Sharpe Ratio Analysis  
    - Weight Dynamics & Portfolio Value  
    - Date Range: *03-06-2024 → Today*  
    - Investment: ₹50,000 per stock  

    Use the sidebar to navigate.
    """)


def optimization_page():
    st.title("🎯 Portfolio Optimization")

    start_date = st.date_input("Start Date", pd.to_datetime(DEFAULT_START))
    end_date = st.date_input("End Date", pd.to_datetime(DEFAULT_END))

    if st.button("🚀 Run Optimization"):
        with st.spinner("Fetching Data..."):
            data = fetch_data(list(STOCKS.keys()), start_date, end_date)

        if data.empty:
            return

        returns = calculate_daily_returns(data)

        st.subheader("Optimal Weights")
        opt_weights = optimize_portfolio(returns)

        df = pd.DataFrame({
            "Stock": list(STOCKS.values()),
            "Ticker": list(STOCKS.keys()),
            "Weight": opt_weights,
            "Weight (%)": opt_weights * 100
        })

        st.dataframe(df, use_container_width=True)

        fig = px.pie(df, names="Ticker", values="Weight (%)",
                     title="Optimized Allocation")
        st.plotly_chart(fig)

        portfolio_df, weights = calculate_portfolio_value(INITIAL_INVESTMENT, data)

        fig2 = px.line(
            x=portfolio_df.index,
            y=portfolio_df["portfolio_value"],
            title="Portfolio Value Over Time"
        )
        st.plotly_chart(fig2)


def analysis_page():
    st.title("📊 Portfolio Analysis")

    start_date = pd.to_datetime(DEFAULT_START)
    end_date = pd.to_datetime(DEFAULT_END)

    with st.spinner("Loading Data..."):
        data = fetch_data(list(STOCKS.keys()), start_date, end_date)

    if data.empty:
        return

    returns = calculate_daily_returns(data)
    portfolio_df, weights = calculate_portfolio_value(INITIAL_INVESTMENT, data)
    daily_sharpe = calculate_daily_sharpe_ratios(returns)

    st.subheader("Daily Sharpe Ratio Δ")
    st.plotly_chart(plot_daily_sharpe_differences(returns))

    st.subheader("Daily Returns")
    st.plotly_chart(px.line(returns, title="Daily Returns of Stocks"))

    st.subheader("Initial Weights (Based on Opening Day Price)")
    st.write(weights)

    st.subheader("Portfolio Value Data")
    st.dataframe(portfolio_df)


# --------------------------------------------------------------------
# MAIN ROUTER
# --------------------------------------------------------------------
pages = {
    "Home": home_page,
    "Portfolio Optimization": optimization_page,
    "Analysis": analysis_page
}

with st.sidebar:
    st.title("🧭 Navigation")
    choice = st.radio("Go to:", list(pages.keys()))

pages[choice]()