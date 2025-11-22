import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import datetime as dt
import time
from scipy.optimize import minimize

# =====================================================
# CONFIG
# =====================================================

API_KEY = "99BMKKVFJD0BVC8E"   # Your AlphaVantage API Key

STOCKS = [
    "HCLTECH.NS",
    "ADANIENT.NS",
    "TECHM.NS",
    "INFY.NS",
    "WIPRO.NS",
    "OFSS.NS",
    "MPHASIS.NS",
    "LTIM.NS",
    "PERSISTENT.NS",
    "TCS.NS",
]

INITIAL_INVESTMENT = 50000  # Per stock
DEFAULT_START = dt.date(2024, 6, 3)


# =====================================================
# DATA FETCHING — ALPHA VANTAGE (GLOBAL SAFE)
# =====================================================

def fetch_alpha_vantage(symbol):
    """
    Fetch daily historical data (full length) from Alpha Vantage.
    Works on Streamlit Cloud 100%.
    """
    url = (
        "https://www.alphavantage.co/query?"
        f"function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&outputsize=full&apikey={API_KEY}"
    )

    r = requests.get(url)
    if r.status_code != 200:
        return pd.DataFrame()

    data = r.json()
    if "Time Series (Daily)" not in data:
        return pd.DataFrame()

    ts = data["Time Series (Daily)"]
    df = pd.DataFrame(ts).T
    df.index = pd.to_datetime(df.index)

    df = df.rename(
        columns={
            "1. open": "Open",
            "2. high": "High",
            "3. low": "Low",
            "4. close": "Close",
        }
    )[["Open", "High", "Low", "Close"]]

    df = df.astype(float)

    return df.sort_index()


@st.cache_data(show_spinner=False)
def fetch_all_stocks(symbols, start_date, end_date):
    """
    Fetch all stocks using Alpha Vantage
    with built-in rate-limit sleep.
    """
    final = pd.DataFrame()

    for sym in symbols:
        st.write(f"Fetching {sym} ...")

        df = fetch_alpha_vantage(sym)
        time.sleep(12)  # Alpha Vantage free plan requires 5 calls/min

        if df.empty:
            st.warning(f"⚠ No data for {sym}")
            continue

        df = df.loc[start_date:end_date]
        final[sym] = df["Open"]

    if final.empty:
        st.error("❌ Could not download ANY stock data.")
        return pd.DataFrame()

    return final.ffill().dropna()


# =====================================================
# PORTFOLIO FUNCTIONS
# =====================================================

def calculate_portfolio_value(initial_investment, data):
    if data.empty:
        return data, pd.Series(dtype=float)

    first_prices = data.iloc[0]
    shares = initial_investment / first_prices
    portfolio_value = (data * shares).sum(axis=1)

    out = data.copy()
    out["portfolio_value"] = portfolio_value
    return out, shares


def calculate_daily_returns(data):
    return data.pct_change().dropna()


def calculate_sharpe_ratio(series, rf=0.0):
    if series.std() == 0:
        return 0
    return (series.mean() - rf) / series.std()


def optimize_portfolio(returns, rf=0.0):
    if returns.empty:
        return np.array([])

    n = len(returns.columns)
    mean_r = returns.mean()
    cov = returns.cov()

    def neg_sharpe(w):
        r = np.dot(w, mean_r)
        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        if vol == 0:
            return 999
        return -(r - rf) / vol

    init = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1}

    res = minimize(neg_sharpe, init, method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else init


# =====================================================
# UI PAGES
# =====================================================

def page_home():
    st.title("📊 Portfolio Analyzer (Alpha Vantage Powered)")
    st.markdown("""
    This app performs **portfolio analysis & optimization** for 10 NSE tech stocks  
    using **free & global Alpha Vantage API** (works on Streamlit Cloud).

    **Features**
    - Robust data fetching (no yfinance errors)
    - Daily returns, Sharpe ratios
    - Portfolio optimization (Mean-Variance)
    - Portfolio value & allocation charts
    - Time range: **2024-06-03 → Today**
    """)


def page_optimization():
    st.title("🎯 Portfolio Optimization")

    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start Date", DEFAULT_START)
    with c2:
        end_date = st.date_input("End Date", dt.date.today())

    if st.button("🚀 Run Optimization"):
        with st.spinner("Fetching stock data..."):
            data = fetch_all_stocks(STOCKS, start_date, end_date)

        if data.empty:
            st.stop()

        returns = calculate_daily_returns(data)
        weights = optimize_portfolio(returns)

        st.subheader("Optimal Portfolio Weights")
        dfw = pd.DataFrame({
            "Stock": STOCKS,
            "Weight": weights,
            "Weight (%)": weights * 100
        })
        st.dataframe(dfw)

        fig = px.pie(dfw, names="Stock", values="Weight", title="Optimal Allocation")
        st.plotly_chart(fig)

        with st.spinner("Calculating portfolio value..."):
            portfolio_df, shares = calculate_portfolio_value(INITIAL_INVESTMENT, data)

        fig2 = px.line(
            portfolio_df,
            x=portfolio_df.index,
            y="portfolio_value",
            title="Portfolio Value (₹)"
        )
        st.plotly_chart(fig2)

        st.subheader("📄 Price Data")
        st.dataframe(portfolio_df)


def page_analysis():
    st.title("📈 Detailed Analysis")

    start_date = DEFAULT_START
    end_date = dt.date.today()

    with st.spinner("Downloading data..."):
        data = fetch_all_stocks(STOCKS, start_date, end_date)

    if data.empty:
        st.stop()

    returns = calculate_daily_returns(data)

    st.subheader("Stock Price Chart")
    st.plotly_chart(px.line(data, title="Price History"))

    st.subheader("Daily Returns (%)")
    st.plotly_chart(px.line(returns, title="Daily Returns"))

    # Sharpe per stock
    sharpe = {s: calculate_sharpe_ratio(returns[s]) for s in returns.columns}
    df_sharpe = pd.DataFrame({
        "Stock": sharpe.keys(),
        "Sharpe Ratio": sharpe.values()
    })

    st.subheader("Sharpe Ratios")
    st.plotly_chart(px.bar(df_sharpe, x="Stock", y="Sharpe Ratio"))


# =====================================================
# NAVIGATION
# =====================================================

PAGES = {
    "Home": page_home,
    "Portfolio Optimization": page_optimization,
    "Analysis": page_analysis,
}

def main():
    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Select Page", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()