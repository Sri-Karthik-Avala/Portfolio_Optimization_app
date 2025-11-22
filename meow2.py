import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import warnings
from datetime import datetime
import yfinance as yf   # Streamlit Cloud supports this

warnings.filterwarnings('ignore')

# -----------------------
# STOCK CONFIG
# -----------------------
STOCKS = {
    'RELIANCE': {'name': 'Reliance Industries', 'symbol': 'RELIANCE.NS'},
    'TCS': {'name': 'Tata Consultancy Services', 'symbol': 'TCS.NS'},
    'INFY': {'name': 'Infosys', 'symbol': 'INFY.NS'},
    'HDFCBANK': {'name': 'HDFC Bank', 'symbol': 'HDFCBANK.NS'},
    'ICICIBANK': {'name': 'ICICI Bank', 'symbol': 'ICICIBANK.NS'},
    'HINDUNILVR': {'name': 'Hindustan Unilever', 'symbol': 'HINDUNILVR.NS'},
    'ITC': {'name': 'ITC Limited', 'symbol': 'ITC.NS'},
    'SBIN': {'name': 'State Bank of India', 'symbol': 'SBIN.NS'},
    'BHARTIARTL': {'name': 'Bharti Airtel', 'symbol': 'BHARTIARTL.NS'},
    'ASIANPAINT': {'name': 'Asian Paints', 'symbol': 'ASIANPAINT.NS'}
}

INITIAL_INVESTMENT = 50000

# -----------------------
# FETCH STOCK DATA
# -----------------------
def fetch_stock_data(start_date, end_date):
    """Fetch stock data from Yahoo Finance."""
    data = pd.DataFrame()
    progress = st.progress(0)

    symbols = [v['symbol'] for v in STOCKS.values()]
    for i, symbol in enumerate(symbols):
        try:
            df = yf.download(symbol, start=start_date, end=end_date)['Close']
            data[symbol] = df
        except Exception as e:
            st.warning(f"Couldn't fetch data for {symbol}: {e}")

        progress.progress((i + 1) / len(symbols))

    progress.empty()

    if data.empty:
        return pd.DataFrame()

    return data.dropna()

# -----------------------
# PORTFOLIO CALCULATIONS
# -----------------------
def calculate_portfolio_metrics(data, initial_investment):
    if data.empty:
        return pd.Series(), pd.Series()

    first_prices = data.iloc[0]
    shares = initial_investment / first_prices
    portfolio_values = (data * shares).sum(axis=1)
    return portfolio_values, shares

def calculate_returns(data):
    return data.pct_change().dropna()

def calculate_sharpe_ratio(returns, rf=0.0):
    if returns.std() == 0:
        return 0
    return (returns.mean() - rf) / returns.std()

def optimize_portfolio(returns):
    if returns.empty:
        return np.array([])

    n = len(returns.columns)
    mean_r = returns.mean()
    cov = returns.cov()

    def neg_sharpe(weights):
        ret = np.dot(weights, mean_r)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        if vol == 0:
            return 1000
        return -(ret / vol)

    initial = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    try:
        res = minimize(neg_sharpe, initial, bounds=bounds, constraints=cons)
        if res.success:
            return res.x
    except:
        pass

    return initial

# -----------------------
# PAGES
# -----------------------
def page_home():
    st.title('📈 Portfolio Analysis Dashboard')
    st.markdown("""
    Welcome to the Indian Stock Portfolio Analyzer!

    **What you get here:**
    - Live stock data
    - Portfolio optimization (Modern Portfolio Theory)
    - Sharpe ratio insights
    - Interactive charts
    """)

def page_optimization():
    st.title("🎯 Portfolio Optimization")

    col1, col2 = st.columns(2)
    start = col1.date_input("Start Date", value=pd.to_datetime("2023-01-01"))
    end = col2.date_input("End Date", value=pd.Timestamp.today())

    if st.button("🚀 Optimize Portfolio"):
        with st.spinner("Fetching data..."):
            data = fetch_stock_data(start, end)

        if data.empty:
            st.error("No data found.")
            return

        returns = calculate_returns(data)
        weights = optimize_portfolio(returns)

        st.subheader("📊 Optimal Portfolio Weights")
        df = pd.DataFrame({
            "Stock": [v['name'] for v in STOCKS.values()],
            "Symbol": list(data.columns),
            "Weight": weights,
            "Weight (%)": weights * 100
        })
        st.dataframe(df, use_container_width=True)

        fig = px.pie(df, values="Weight (%)", names="Symbol", title="Optimal Allocation")
        st.plotly_chart(fig)

        # Portfolio performance
        portfolio_values, _ = calculate_portfolio_metrics(data, INITIAL_INVESTMENT)

        fig_line = px.line(
            x=portfolio_values.index,
            y=portfolio_values.values,
            title='Portfolio Value Over Time'
        )
        st.plotly_chart(fig_line)

        st.metric("Initial Value", f"₹{portfolio_values.iloc[0]:,.0f}")
        st.metric("Final Value", f"₹{portfolio_values.iloc[-1]:,.0f}")
        st.metric("Total Return", f"{((portfolio_values.iloc[-1]/portfolio_values.iloc[0])-1)*100:.2f}%")

def page_analysis():
    st.title("📊 Portfolio Analysis")

    col1, col2 = st.columns(2)
    start = col1.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end = col2.date_input("End Date", pd.Timestamp.today())

    if st.button("📈 Run Analysis"):
        with st.spinner("Fetching..."):
            data = fetch_stock_data(start, end)

        if data.empty:
            st.error("No data available.")
            return

        returns = calculate_returns(data)

        st.subheader("📈 Price Chart")
        st.plotly_chart(px.line(data, title="Stock Prices"))

        st.subheader("📊 Daily Returns")
        st.plotly_chart(px.line(returns, title="Daily Returns"))

        st.subheader("⚡ Sharpe Ratios")
        sharpe = {col: calculate_sharpe_ratio(returns[col]) for col in returns.columns}

        df = pd.DataFrame({
            "Stock": [STOCKS[s]['name'] for s in sharpe.keys()],
            "Symbol": list(sharpe.keys()),
            "Sharpe Ratio": list(sharpe.values())
        })

        st.plotly_chart(px.bar(df, x="Symbol", y="Sharpe Ratio", title="Sharpe Ratios"))

# -----------------------
# MAIN
# -----------------------
def main():
    st.set_page_config(page_title="Portfolio Analysis", page_icon="📈", layout="wide")

    page = st.sidebar.radio("Navigate", ["Homepage", "Portfolio Optimization", "Analysis"])

    if page == "Homepage":
        page_home()
    elif page == "Portfolio Optimization":
        page_optimization()
    else:
        page_analysis()

if __name__ == '__main__':
    main()