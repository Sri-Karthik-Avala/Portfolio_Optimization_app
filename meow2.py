import os
import time
from datetime import datetime, date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
from scipy.optimize import minimize

# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Portfolio Analysis - Mean Variance",
    page_icon="📈",
    layout="wide",
)

# Alpha Vantage API key (you gave this)
# You can also override via Streamlit secrets or env var later.
ALPHA_VANTAGE_API_KEY = os.getenv(
    "ALPHAVANTAGE_API_KEY",
    "99BMKKVFJD0BVC8E"
)

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"

# Start and end date for the project (fixed as per your report)
DEFAULT_START_DATE = date(2024, 6, 3)
DEFAULT_END_DATE = date.today()

# Initial investment PER STOCK (as per your report)
INITIAL_INVESTMENT_PER_STOCK = 50_000  # INR

# Tech stocks from your IBA report
ALL_STOCKS = {
    "HCLTECH.NS": "HCL Technologies",
    "ADANIENT.NS": "Adani Enterprises",
    "TECHM.NS": "Tech Mahindra",
    "INFY.NS": "Infosys",
    "WIPRO.NS": "Wipro",
    "OFSS.NS": "Oracle Financial Services",
    "MPHASIS.NS": "Mphasis",
    "LTIM.NS": "LTIMindtree",
    "PERSISTENT.NS": "Persistent Systems",
    "TCS.NS": "Tata Consultancy Services",
}


# ------------------------------------------------------------------------------
# DATA FETCHING (Alpha Vantage)
# ------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_stock_data(symbols, start_date, end_date):
    """
    Fetch daily OPEN prices for the given symbols from Alpha Vantage.
    Returns a DataFrame with Date index and columns = symbols.

    If some symbols fail, they are skipped. If all fail, returns empty DF.
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.error("Alpha Vantage API key missing. Set ALPHAVANTAGE_API_KEY or edit the code.")
        return pd.DataFrame()

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    all_series = {}
    errors = []

    for i, symbol in enumerate(symbols, start=1):
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "apikey": ALPHA_VANTAGE_API_KEY,
                "outputsize": "full",
                "datatype": "json",
            }
            resp = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data_json = resp.json()

            if "Time Series (Daily)" not in data_json:
                msg = data_json.get("Note") or data_json.get("Error Message") or "Unknown API response"
                errors.append(f"{symbol}: {msg}")
                continue

            ts = data_json["Time Series (Daily)"]
            df = pd.DataFrame.from_dict(ts, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)

            # Use the "1. open" price
            series = df["1. open"]
            # Filter by date range
            series = series.loc[(series.index >= start_date) & (series.index <= end_date)]

            if series.empty:
                errors.append(f"{symbol}: no data returned in selected date range.")
                continue

            all_series[symbol] = series

        except Exception as e:
            errors.append(f"{symbol}: {e}")

        # Alpha Vantage free tier: 5 calls/minute -> be gentle
        # If user selects many stocks, they may still hit the limit,
        # but this sleep helps.
        time.sleep(12)

    if errors:
        st.warning(
            "Some symbols could not be fetched:\n\n" +
            "\n".join([f"- {e}" for e in errors])
        )

    if not all_series:
        return pd.DataFrame()

    data = pd.DataFrame(all_series)
    data = data.sort_index()
    data = data.ffill().dropna(how="all")

    return data


# ------------------------------------------------------------------------------
# FINANCIAL FUNCTIONS
# ------------------------------------------------------------------------------

def calculate_portfolio_value(initial_investment_per_stock, data):
    """
    Calculate portfolio value over time assuming equal rupee investment
    in each stock at the first available date.

    Returns:
    - data_with_portfolio: DataFrame with extra column 'portfolio_value'
    - weights: Series with weights (value per stock / total)
    """
    if data.empty:
        return data, pd.Series(dtype=float)

    # total rupees invested per stock (constant)
    # but we calculate "shares" using first day's price
    first_prices = data.iloc[0]
    investment_per_stock = initial_investment_per_stock
    shares = investment_per_stock / first_prices  # number of shares per stock

    # portfolio value each day = sum(price * shares)
    portfolio_values = (data * shares).sum(axis=1)
    data_with_portfolio = data.copy()
    data_with_portfolio["portfolio_value"] = portfolio_values

    # weights expressed as fraction of total initial capital
    total_initial_value = portfolio_values.iloc[0]
    weights = (first_prices * shares) / total_initial_value

    return data_with_portfolio, weights


def calculate_daily_returns(data):
    """Calculate daily returns (pct_change) for each stock."""
    return data.pct_change(1).dropna(how="all")


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio given a returns Series."""
    if returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    return float((returns.mean() - risk_free_rate) / returns.std())


def calculate_daily_sharpe_ratios(returns, risk_free_rate=0.0):
    """
    Calculate rolling Sharpe ratio up to each date, for each stock.
    Returns a DataFrame with same shape as returns.
    """
    if returns.empty:
        return pd.DataFrame()

    sharpe_ratios = pd.DataFrame(index=returns.index, columns=returns.columns)

    for i in range(1, len(returns)):
        sub = returns.iloc[:i]
        sharpe_ratios.iloc[i] = sub.apply(calculate_sharpe_ratio, axis=0, risk_free_rate=risk_free_rate)

    return sharpe_ratios


def plot_daily_sharpe_differences(returns, risk_free_rate=0.0):
    """
    Plot daily differences in portfolio Sharpe ratio over time.
    Here we treat portfolio as equal-weight returns (for visualization).
    """
    if returns.empty:
        return px.line(title="Daily Differences in Sharpe Ratios (no data)")

    # equal-weight portfolio
    ew_returns = returns.mean(axis=1)
    sharpe_series = []
    for i in range(1, len(ew_returns)):
        sub = ew_returns.iloc[:i]
        sr = calculate_sharpe_ratio(sub, risk_free_rate)
        sharpe_series.append(sr)

    # align indexes
    idx = ew_returns.index[1:]
    sharpe_series = pd.Series(sharpe_series, index=idx)
    sharpe_diff = sharpe_series.diff()

    fig = px.line(
        x=sharpe_diff.index,
        y=sharpe_diff.values,
        title="Daily Differences in Sharpe Ratios (Equal-Weight Portfolio)",
        labels={"x": "Date", "y": "Δ Sharpe Ratio"}
    )
    return fig


def plot_portfolio_weights(weights):
    """Plot static initial weights (bar chart)."""
    if weights.empty:
        return px.bar(title="Portfolio Weights (no data)")

    df = pd.DataFrame({
        "Symbol": weights.index,
        "Weight": weights.values
    })

    fig = px.bar(
        df,
        x="Symbol",
        y="Weight",
        title="Initial Portfolio Weights",
        labels={"Weight": "Weight (fraction of total)"},
    )
    return fig


def plot_daily_weight_changes(data, weights):
    """
    Approximate daily change in value allocation as:
    weight_t = (price_t * shares) / portfolio_value_t
    """
    if data.empty or weights.empty:
        return px.line(title="Daily Weight Changes (%) (no data)")

    price_data = data[weights.index]
    first_prices = price_data.iloc[0]
    shares = (weights * (price_data.iloc[0] * 1)) / first_prices  # derive shares ~ proportional

    portfolio_values = (price_data * shares).sum(axis=1)
    alloc = (price_data.mul(shares, axis=1).div(portfolio_values, axis=0))  # weights each day
    alloc_change_pct = alloc.pct_change() * 100

    fig = px.line(
        alloc_change_pct,
        x=alloc_change_pct.index,
        y=list(alloc_change_pct.columns),
        title="Daily Change in Allocation (%)",
        labels={"index": "Date", "value": "Change in weight (%)"},
    )
    return fig


def optimize_portfolio(returns, risk_free_rate=0.0):
    """
    Mean-variance optimization: maximize Sharpe ratio
    => minimize negative Sharpe.
    """
    if returns.empty:
        return np.array([])

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(mean_returns)

    def neg_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_std == 0:
            return 1e6
        return -(portfolio_return - risk_free_rate) / portfolio_std

    bounds = tuple((0, 1) for _ in range(num_assets))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    init_guess = np.array(num_assets * [1.0 / num_assets])

    try:
        result = minimize(
            neg_sharpe,
            init_guess,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if not result.success:
            return init_guess
        return result.x
    except Exception:
        return init_guess


# ------------------------------------------------------------------------------
# PAGES
# ------------------------------------------------------------------------------

def page_home():
    st.title("📈 Investment Optimization: Mean-Variance Analysis")
    st.markdown(
        """
        ### Welcome to the Portfolio Analysis Application
        
        This app is rebuilt to match your **IBA project report**:
        
        - Uses **10 Indian tech companies (NSE)**  
        - Fetches data from **2024-06-03 till today** (configurable in code)  
        - Performs **Mean-Variance Optimization** to maximize the Sharpe ratio  
        - Provides:
          - Portfolio optimization  
          - Daily returns & Sharpe ratios  
          - Portfolio value over time  
          - Daily changes in portfolio weights
        
        Use the sidebar to switch between:
        - 🏠 Home  
        - 🎯 Portfolio Optimization  
        - 📊 Analysis
        """
    )


def page_portfolio_optimization():
    st.title("🎯 Portfolio Optimization")

    st.caption(
        f"Data range (fixed for project): **{DEFAULT_START_DATE} → {DEFAULT_END_DATE}**"
    )

    # Let user choose subset of stocks (helps with free API limits)
    symbols = st.multiselect(
        "Select stocks to include:",
        options=list(ALL_STOCKS.keys()),
        default=list(ALL_STOCKS.keys()),
        format_func=lambda s: f"{s} - {ALL_STOCKS[s]}",
    )

    if not symbols:
        st.warning("Please select at least one stock.")
        return

    if st.button("🚀 Run Optimization"):
        with st.spinner("Fetching data & optimizing portfolio..."):
            data = fetch_stock_data(symbols, DEFAULT_START_DATE, DEFAULT_END_DATE)

            if data.empty:
                st.error(
                    "No stock data downloaded. This is usually due to the external API "
                    "rate limit or network issues. Try again later or with fewer stocks."
                )
                return

            returns = calculate_daily_returns(data)

            if returns.empty:
                st.error("Could not compute returns. Check data.")
                return

            opt_weights = optimize_portfolio(returns)

            st.subheader("📊 Optimal Portfolio Weights")
            weights_df = pd.DataFrame(
                {
                    "Symbol": data.columns,
                    "Company": [ALL_STOCKS[s] for s in data.columns],
                    "Optimal Weight": opt_weights,
                    "Optimal Weight (%)": opt_weights * 100,
                }
            ).set_index("Symbol")
            st.dataframe(weights_df.style.format({"Optimal Weight (%)": "{:.2f}"}), use_container_width=True)

            # Pie chart of optimal weights
            fig_pie = px.pie(
                weights_df.reset_index(),
                values="Optimal Weight",
                names="Symbol",
                title="Optimal Allocation by Symbol",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Portfolio value using equal rupee investment per stock (50k each)
            data_with_portfolio, static_weights = calculate_portfolio_value(
                INITIAL_INVESTMENT_PER_STOCK, data
            )

            if "portfolio_value" in data_with_portfolio.columns:
                st.subheader("💰 Portfolio Value Over Time")
                fig_val = px.line(
                    data_with_portfolio,
                    x=data_with_portfolio.index,
                    y="portfolio_value",
                    title="Portfolio Value Over Time",
                    labels={"index": "Date", "portfolio_value": "Portfolio Value (₹)"},
                )
                st.plotly_chart(fig_val, use_container_width=True)

                initial_val = data_with_portfolio["portfolio_value"].iloc[0]
                final_val = data_with_portfolio["portfolio_value"].iloc[-1]
                total_return = (final_val - initial_val) / initial_val * 100

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Initial Portfolio Value", f"₹{initial_val:,.0f}")
                with col2:
                    st.metric("Final Portfolio Value", f"₹{final_val:,.0f}")
                with col3:
                    st.metric("Total Return (%)", f"{total_return:.2f}%")

            st.subheader("📌 Static Initial Weights (Equal Rupee per Stock)")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Symbol": static_weights.index,
                        "Company": [ALL_STOCKS[s] for s in static_weights.index],
                        "Weight": static_weights.values,
                    }
                ).set_index("Symbol"),
                use_container_width=True,
            )


def page_analysis():
    st.title("📊 Detailed Portfolio Analysis")

    st.caption(
        f"Data range (fixed for project): **{DEFAULT_START_DATE} → {DEFAULT_END_DATE}**"
    )

    symbols = st.multiselect(
        "Select stocks to analyse:",
        options=list(ALL_STOCKS.keys()),
        default=list(ALL_STOCKS.keys()),
        format_func=lambda s: f"{s} - {ALL_STOCKS[s]}",
        key="analysis_symbols",
    )

    if not symbols:
        st.warning("Please select at least one stock.")
        return

    if st.button("📈 Run Analysis"):
        with st.spinner("Fetching data & running analysis..."):
            data = fetch_stock_data(symbols, DEFAULT_START_DATE, DEFAULT_END_DATE)

            if data.empty:
                st.error(
                    "No stock data downloaded. This is usually due to the external API "
                    "rate limit or network issues. Try again later or with fewer stocks."
                )
                return

            returns = calculate_daily_returns(data)

            st.subheader("📈 Stock Prices Over Time")
            fig_prices = px.line(
                data,
                x=data.index,
                y=list(data.columns),
                title="Stock Prices (Open) Over Time",
                labels={"index": "Date", "value": "Price (₹)"},
            )
            st.plotly_chart(fig_prices, use_container_width=True)

            st.subheader("📊 Daily Returns")
            fig_returns = px.line(
                returns,
                x=returns.index,
                y=list(returns.columns),
                title="Daily Returns",
                labels={"index": "Date", "value": "Return"},
            )
            st.plotly_chart(fig_returns, use_container_width=True)

            # Sharpe ratios per stock
            st.subheader("⚡ Sharpe Ratios by Stock (Full Period)")
            sharpe_ratios = {
                s: calculate_sharpe_ratio(returns[s].dropna()) for s in returns.columns
            }
            sharpe_df = pd.DataFrame(
                {
                    "Symbol": list(sharpe_ratios.keys()),
                    "Company": [ALL_STOCKS[s] for s in sharpe_ratios.keys()],
                    "Sharpe Ratio": list(sharpe_ratios.values()),
                }
            ).sort_values("Sharpe Ratio", ascending=False)

            fig_sharpe = px.bar(
                sharpe_df,
                x="Symbol",
                y="Sharpe Ratio",
                title="Sharpe Ratios by Stock",
                color="Sharpe Ratio",
                color_continuous_scale="viridis",
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)

            st.dataframe(sharpe_df.set_index("Symbol"), use_container_width=True)

            # Portfolio value & weights
            data_with_portfolio, weights = calculate_portfolio_value(
                INITIAL_INVESTMENT_PER_STOCK, data
            )

            st.subheader("💰 Portfolio Value Over Time")
            fig_port_val = px.line(
                data_with_portfolio,
                x=data_with_portfolio.index,
                y="portfolio_value",
                title="Portfolio Value Over Time",
                labels={"index": "Date", "portfolio_value": "Portfolio Value (₹)"},
            )
            st.plotly_chart(fig_port_val, use_container_width=True)

            st.subheader("📌 Initial Stock Weights (Equal Rupee per Stock)")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Symbol": weights.index,
                        "Company": [ALL_STOCKS[s] for s in weights.index],
                        "Weight": weights.values,
                    }
                ).set_index("Symbol"),
                use_container_width=True,
            )

            # Daily Sharpe diffs
            st.subheader("📉 Daily Differences in Sharpe Ratios (Equal-Weight Portfolio)")
            fig_sharpe_diff = plot_daily_sharpe_differences(returns)
            st.plotly_chart(fig_sharpe_diff, use_container_width=True)

            # Weight charts
            st.subheader("📊 Initial Portfolio Weights (Bar)")
            fig_weights = plot_portfolio_weights(weights)
            st.plotly_chart(fig_weights, use_container_width=True)

            st.subheader("📉 Daily Weight Changes (%) (Approx.)")
            fig_weight_changes = plot_daily_weight_changes(data_with_portfolio, weights)
            st.plotly_chart(fig_weight_changes, use_container_width=True)

            st.subheader("📋 Summary Statistics")
            summary_stats = pd.DataFrame(
                {
                    "Symbol": returns.columns,
                    "Company": [ALL_STOCKS[s] for s in returns.columns],
                    "Mean Daily Return (%)": (returns.mean() * 100).values,
                    "Volatility (%)": (returns.std() * 100).values,
                    "Sharpe Ratio": [sharpe_ratios[s] for s in returns.columns],
                }
            ).set_index("Symbol")
            st.dataframe(summary_stats.round(4), use_container_width=True)


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------

def main():
    st.sidebar.title("🧭 Navigation")