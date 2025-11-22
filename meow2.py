import datetime as dt

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import minimize

try:
    import yfinance as yf
except ImportError:
    yf = None

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Default start date: 2024-06-03 (as per your project)
DEFAULT_START_DATE = dt.date(2024, 6, 3)

# List of Indian tech/IT stocks from the report, NSE tickers
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

# Initial investment PER STOCK (₹), so total portfolio is 10 * 50,000 = 5,00,000
INITIAL_INVESTMENT_PER_STOCK = 50_000


# ------------------------------------------------------------
# DATA FETCHING
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_data(stocks, start_date, end_date):
    """
    Fetch daily 'Open' prices for each ticker using yfinance.

    Returns:
        pd.DataFrame with Date index and one column per stock.
    """
    if yf is None:
        st.error("yfinance is not installed. Please add `yfinance` to requirements.txt.")
        return pd.DataFrame()

    # Ensure we have proper date objects
    if isinstance(start_date, dt.date):
        start = start_date
    else:
        start = pd.to_datetime(start_date).date()

    if isinstance(end_date, dt.date):
        end = end_date
    else:
        end = pd.to_datetime(end_date).date()

    data = pd.DataFrame()
    successful = []

    for ticker in stocks:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end + dt.timedelta(days=1),  # include end date
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
        except Exception as e:
            st.warning(f"❌ Could not fetch: {ticker} (error: {e})")
            continue

        if df is None or df.empty:
            st.warning(f"❌ No data for: {ticker} in this date range.")
            continue

        # Use 'Open' prices
        data[ticker] = df["Open"].astype(float)
        successful.append(ticker)

    if data.empty:
        st.error(
            "No stock data downloaded. "
            "This is usually due to Yahoo Finance being temporarily unavailable "
            "or the date range being outside trading days."
        )
        return pd.DataFrame()

    data = data.sort_index()
    data = data.ffill().dropna(how="all")

    st.success(f"✅ Downloaded data for {len(successful)} stocks.")
    return data


# ------------------------------------------------------------
# CALCULATIONS
# ------------------------------------------------------------

def calculate_portfolio_value(initial_investment_per_stock, data):
    """
    Calculate portfolio value over time and the initial number of shares per stock.

    initial_investment_per_stock: rupees invested in EACH stock on day 1.
    """
    if data.empty:
        return data, pd.Series(dtype=float)

    first_prices = data.iloc[0]
    # Number of shares for each stock at t=0
    weights = initial_investment_per_stock / first_prices
    # Total portfolio value = Σ (shares_i * price_i_t)
    portfolio_value = (data * weights).sum(axis=1)
    data_with_portfolio = data.copy()
    data_with_portfolio["portfolio_value"] = portfolio_value

    return data_with_portfolio, weights


def calculate_daily_returns(data):
    """Simple daily percentage returns."""
    return data.pct_change(1).dropna(how="all")


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Risk-adjusted return per unit volatility."""
    if returns.std() == 0 or np.isnan(returns.std()):
        return 0.0
    return (returns.mean() - risk_free_rate) / returns.std()


def calculate_daily_sharpe_ratios(returns, risk_free_rate=0.0):
    """Compute rolling Sharpe ratio for each stock."""
    if returns.empty:
        return pd.DataFrame()

    sharpe_ratios = pd.DataFrame(index=returns.index, columns=returns.columns)
    for i in range(1, len(returns)):
        sub = returns.iloc[:i]
        sharpe_ratios.iloc[i] = sub.apply(
            calculate_sharpe_ratio, axis=0, risk_free_rate=risk_free_rate
        )
    return sharpe_ratios


def optimize_portfolio(returns, risk_free_rate=0.0):
    """Mean–variance optimization to maximise Sharpe ratio."""
    if returns.empty:
        return np.array([])

    num_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, rf):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if portfolio_std == 0:
            return 1e6
        return -(portfolio_return - rf) / portfolio_std

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = np.array(num_assets * [1.0 / num_assets])

    try:
        opt_result = minimize(
            neg_sharpe_ratio,
            init_guess,
            args=(mean_returns, cov_matrix, risk_free_rate),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if opt_result.success:
            return opt_result.x
        else:
            return init_guess
    except Exception:
        return init_guess


# ------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------

def plot_daily_sharpe_differences(returns, rf=0.0):
    daily_sharpe = returns.apply(calculate_sharpe_ratio, axis=1, risk_free_rate=rf)
    daily_sharpe_diff = daily_sharpe.diff()
    fig = px.line(
        x=daily_sharpe_diff.index,
        y=daily_sharpe_diff.values,
        title="Daily Differences in Sharpe Ratios",
        labels={"x": "Date", "y": "Δ Sharpe ratio"},
    )
    return fig


def plot_portfolio_weights(data, weights):
    """Plot *value based* weights over time."""
    stock_cols = [c for c in data.columns if c in STOCKS]
    if len(stock_cols) == 0:
        return None

    # Value of each stock each day
    value_per_stock = data[stock_cols].multiply(weights[stock_cols], axis=1)
    total_value = value_per_stock.sum(axis=1)
    weights_over_time = value_per_stock.div(total_value, axis=0)

    fig = px.line(
        weights_over_time,
        x=weights_over_time.index,
        y=weights_over_time.columns,
        title="Portfolio Weights Over Time",
        labels={"index": "Date", "value": "Weight"},
    )
    return fig


def plot_daily_weight_changes(data, weights):
    """Daily % change in value based weights."""
    stock_cols = [c for c in data.columns if c in STOCKS]
    if len(stock_cols) == 0:
        return None

    value_per_stock = data[stock_cols].multiply(weights[stock_cols], axis=1)
    total_value = value_per_stock.sum(axis=1)
    weights_over_time = value_per_stock.div(total_value, axis=0)
    daily_changes = weights_over_time.pct_change().dropna(how="all")

    fig = px.line(
        daily_changes,
        x=daily_changes.index,
        y=daily_changes.columns,
        title="Daily Weight Changes (%)",
        labels={"index": "Date", "value": "Daily weight change (%)"},
    )
    return fig


# ------------------------------------------------------------
# PAGES
# ------------------------------------------------------------

def page_home():
    st.title("📊 Investment Optimization: Mean-Variance Analysis")
    st.markdown(
        """
        This app implements a **portfolio analysis & optimization** workflow
        for a basket of Indian IT/tech stocks.

        **What you can do here:**
        - Download historical data (from **2024-06-03** onwards).
        - Compute **daily returns** and **Sharpe ratios**.
        - Run **mean–variance optimization** to maximise Sharpe ratio.
        - Visualise portfolio value, weights, and risk–return behaviour.
        """
    )
    st.info(
        "Data source: Yahoo Finance via the `yfinance` library. "
        "Sometimes Yahoo is slow or temporarily unavailable – in that case, "
        "you will see warnings per ticker."
    )


def page_portfolio_optimization():
    st.title("🎯 Portfolio Optimization")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start date",
            value=DEFAULT_START_DATE,
            min_value=DEFAULT_START_DATE,
        )
    with col2:
        end_date = st.date_input(
            "End date",
            value=dt.date.today(),
            min_value=start_date,
        )

    st.caption(
        f"Data will be fetched from **{start_date}** to **{end_date}** "
        "(trading days only)."
    )

    if st.button("🚀 Run Optimization", type="primary"):
        with st.spinner("Downloading data and optimising portfolio..."):
            data = fetch_data(STOCKS, start_date, end_date)

            if data.empty:
                st.stop()

            returns = calculate_daily_returns(data)

            if returns.empty:
                st.error("Not enough data to compute returns. Try a wider date range.")
                st.stop()

            opt_weights = optimize_portfolio(returns)

            # Show optimal weights
            st.subheader("Optimal Portfolio Weights")
            weights_df = pd.DataFrame(
                {
                    "Ticker": data.columns,
                    "Weight": opt_weights,
                    "Weight (%)": opt_weights * 100,
                }
            ).set_index("Ticker")
            st.dataframe(weights_df.style.format({"Weight": "{:.4f}", "Weight (%)": "{:.2f}"}))

            # Portfolio value over time using equal investment per stock (from report)
            portfolio_data, shares = calculate_portfolio_value(
                INITIAL_INVESTMENT_PER_STOCK, data
            )

            if "portfolio_value" in portfolio_data.columns:
                fig = px.line(
                    portfolio_data,
                    x=portfolio_data.index,
                    y="portfolio_value",
                    title="Portfolio Value Over Time (₹)",
                    labels={"index": "Date", "portfolio_value": "Portfolio value (₹)"},
                )
                st.plotly_chart(fig, use_container_width=True)

                initial_val = float(portfolio_data["portfolio_value"].iloc[0])
                final_val = float(portfolio_data["portfolio_value"].iloc[-1])
                total_return = (final_val - initial_val) / initial_val * 100

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Initial Value", f"₹{initial_val:,.0f}")
                with c2:
                    st.metric("Final Value", f"₹{final_val:,.0f}")
                with c3:
                    st.metric("Total Return", f"{total_return:.2f}%")

            st.subheader("Raw Price Data")
            st.dataframe(portfolio_data, use_container_width=True)


def page_analysis():
    st.title("📈 Detailed Analysis")

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Analysis start date",
            value=DEFAULT_START_DATE,
            min_value=DEFAULT_START_DATE,
            key="analysis_start",
        )
    with col2:
        end_date = st.date_input(
            "Analysis end date",
            value=dt.date.today(),
            min_value=start_date,
            key="analysis_end",
        )

    st.caption(
        f"Analysis period: **{start_date}** to **{end_date}** "
        "(based on available trading days)."
    )

    if st.button("📊 Run Analysis", type="primary"):
        with st.spinner("Downloading data and running analysis..."):
            data = fetch_data(STOCKS, start_date, end_date)
            if data.empty:
                st.stop()

            returns = calculate_daily_returns(data)
            if returns.empty:
                st.error("Not enough data to compute returns. Try a wider date range.")
                st.stop()

            # Portfolio value & weights
            portfolio_data, shares = calculate_portfolio_value(
                INITIAL_INVESTMENT_PER_STOCK, data
            )

            # ---------- Price chart ----------
            st.subheader("Stock Prices Over Time")
            fig_prices = px.line(
                data,
                x=data.index,
                y=data.columns,
                title="Stock Prices (Open)",
                labels={"index": "Date", "value": "Price (₹)"},
            )
            st.plotly_chart(fig_prices, use_container_width=True)

            # ---------- Daily returns ----------
            st.subheader("Daily Returns")
            fig_returns = px.line(
                returns,
                x=returns.index,
                y=returns.columns,
                title="Daily Returns",
                labels={"index": "Date", "value": "Return"},
            )
            st.plotly_chart(fig_returns, use_container_width=True)

            # ---------- Sharpe ratios (per stock) ----------
            st.subheader("Sharpe Ratios by Stock")
            sharpe_ratios = {
                ticker: calculate_sharpe_ratio(returns[ticker])
                for ticker in returns.columns
            }
            sharpe_df = (
                pd.DataFrame(
                    {
                        "Ticker": list(sharpe_ratios.keys()),
                        "Sharpe Ratio": list(sharpe_ratios.values()),
                    }
                )
                .sort_values("Sharpe Ratio", ascending=False)
                .reset_index(drop=True)
            )

            fig_sharpe = px.bar(
                sharpe_df,
                x="Ticker",
                y="Sharpe Ratio",
                title="Sharpe Ratios (overall, per stock)",
            )
            st.plotly_chart(fig_sharpe, use_container_width=True)
            st.dataframe(sharpe_df, use_container_width=True)

            # ---------- Daily Sharpe differences ----------
            st.subheader("Daily Differences in Portfolio Sharpe Ratio")
            fig_sharpe_diff = plot_daily_sharpe_differences(returns)
            st.plotly_chart(fig_sharpe_diff, use_container_width=True)

            # ---------- Weights over time ----------
            if not shares.empty:
                st.subheader("Portfolio Weights Over Time")
                fig_weights = plot_portfolio_weights(portfolio_data, shares)
                if fig_weights is not None:
                    st.plotly_chart(fig_weights, use_container_width=True)

                st.subheader("Daily Weight Changes (%)")
                fig_weight_changes = plot_daily_weight_changes(portfolio_data, shares)
                if fig_weight_changes is not None:
                    st.plotly_chart(fig_weight_changes, use_container_width=True)

                # Initial rupee allocation
                st.subheader("Initial Stock Allocation (₹)")
                initial_allocation = shares * data.iloc[0]
                weights_numeric = initial_allocation / initial_allocation.sum()

                initial_df = pd.DataFrame(
                    {
                        "Ticker": initial_allocation.index,
                        "Initial Allocation (₹)": initial_allocation.values,
                        "Initial Weight": weights_numeric.values,
                    }
                ).set_index("Ticker")
                st.dataframe(
                    initial_df.style.format(
                        {
                            "Initial Allocation (₹)": "₹{:,.0f}",
                            "Initial Weight": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                )

            # ---------- Extra raw tables ----------
            st.subheader("Returns Data")
            st.dataframe(returns, use_container_width=True)

            st.subheader("Price Data (with portfolio_value)")
            st.dataframe(portfolio_data, use_container_width=True)


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Portfolio Analysis & Optimization",
        page_icon="📈",
        layout="wide",
    )

    pages = {
        "Home": page_home,
        "Portfolio Optimization": page_portfolio_optimization,
        "Analysis": page_analysis,
    }

    st.sidebar.title("🧭 Navigation")
    choice = st.sidebar.radio("Go to", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()