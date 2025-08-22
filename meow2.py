import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import warnings
from datetime import datetime, timedelta

# Import NSE and Indian stock libraries
try:
    from nsepython import *
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False
    st.error("NSEPython not available. Install with: pip install nsepython")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')

# Stock configuration
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

STOCKS))
    
    progress_bar.empty()
    
    if not data.empty:
        data = data.dropna()
        st.success(f"Successfully fetched data for {len(successful_stocks)} stocks")
    
    return data

def calculate_portfolio_metrics(data, initial_investment):
    """Calculate portfolio weights and values"""
    if data.empty:
        return pd.DataFrame(), pd.Series()
    
    # Calculate equal-weight shares based on initial prices
    first_prices = data.iloc[0]
    shares_per_stock = initial_investment / first_prices
    
    # Calculate daily portfolio value
    portfolio_values = (data * shares_per_stock).sum(axis=1)
    
    return portfolio_values, shares_per_stock

def calculate_returns(data):
    """Calculate daily returns"""
    return data.pct_change().dropna()

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe ratio"""
    if returns.std() == 0:
        return 0
    return (returns.mean() - risk_free_rate) / returns.std()

def optimize_portfolio(returns):
    """Optimize portfolio using mean-variance optimization"""
    if returns.empty:
        return np.array([])
    
    n_assets = len(returns.columns)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    def negative_sharpe(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_var = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_var)
        
        if portfolio_std == 0:
            return 1000
        
        sharpe = portfolio_return / portfolio_std
        return -sharpe
    
    # Constraints and bounds
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    try:
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            return initial_weights
            
    except Exception:
        return initial_weights

def page_home():
    st.title('📈 Portfolio Analysis Dashboard')
    st.markdown("""
    ### Welcome to the Indian Stock Portfolio Analyzer
    
    This application helps you analyze and optimize your investment portfolio using top Indian stocks.
    
    **Features:**
    - Portfolio optimization using Modern Portfolio Theory
    - Real-time stock data from Yahoo Finance
    - Interactive charts and analysis
    - Sharpe ratio calculations
    
    **Current Portfolio:** Top 10 Indian stocks including Reliance, TCS, Infosys, HDFC Bank, and more.
    
    Use the sidebar to navigate between sections.
    """)

def page_optimization():
    st.title('🎯 Portfolio Optimization')
    
    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2023-01-01'))
    with col2:
        end_date = st.date_input('End Date', value=pd.Timestamp.today())
    
    if st.button('🚀 Optimize Portfolio', type='primary'):
        with st.spinner('Fetching data and optimizing...'):
            # Fetch data
            data = fetch_stock_data(start_date, end_date)
            
            if data.empty:
                st.error("No data available for the selected date range.")
                return
            
            # Calculate returns
            returns = calculate_returns(data)
            
            # Optimize portfolio
            optimal_weights = optimize_portfolio(returns)
            
            if len(optimal_weights) > 0:
                # Display results
                st.subheader('📊 Optimal Portfolio Weights')
                
                weights_df = pd.DataFrame({
                    'Stock': [STOCKS[stock]['name'] for stock in data.columns],
                    'Symbol': list(data.columns),
                    'Weight': optimal_weights,
                    'Weight (%)': optimal_weights * 100
                }).round(4)
                
                st.dataframe(weights_df, use_container_width=True)
                
                # Pie chart of weights
                fig_pie = px.pie(
                    weights_df, 
                    values='Weight (%)', 
                    names='Symbol',
                    title='Optimal Portfolio Allocation'
                )
                st.plotly_chart(fig_pie)
                
                # Portfolio performance
                portfolio_values, shares = calculate_portfolio_metrics(data, INITIAL_INVESTMENT)
                
                if not portfolio_values.empty:
                    fig_performance = px.line(
                        x=portfolio_values.index,
                        y=portfolio_values.values,
                        title='Portfolio Performance Over Time',
                        labels={'x': 'Date', 'y': 'Portfolio Value (₹)'}
                    )
                    st.plotly_chart(fig_performance)
                    
                    # Performance metrics
                    initial_value = portfolio_values.iloc[0]
                    final_value = portfolio_values.iloc[-1]
                    total_return = ((final_value - initial_value) / initial_value) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric('Initial Value', f'₹{initial_value:,.0f}')
                    with col2:
                        st.metric('Final Value', f'₹{final_value:,.0f}')
                    with col3:
                        st.metric('Total Return', f'{total_return:.2f}%')

def page_analysis():
    st.title('📈 Portfolio Analysis')
    
    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Analysis Start Date', value=pd.to_datetime('2023-01-01'))
    with col2:
        end_date = st.date_input('Analysis End Date', value=pd.Timestamp.today())
    
    if st.button('📊 Run Analysis', type='primary'):
        with st.spinner('Analyzing portfolio...'):
            # Fetch data
            data = fetch_stock_data(start_date, end_date)
            
            if data.empty:
                st.error("No data available for analysis.")
                return
            
            # Calculate returns
            returns = calculate_returns(data)
            
            # Stock prices chart
            st.subheader('📈 Stock Prices Over Time')
            fig_prices = px.line(
                data,
                title='Stock Prices',
                labels={'index': 'Date', 'value': 'Price (₹)'}
            )
            st.plotly_chart(fig_prices)
            
            # Daily returns chart
            st.subheader('📊 Daily Returns')
            fig_returns = px.line(
                returns,
                title='Daily Returns (%)',
                labels={'index': 'Date', 'value': 'Return (%)'}
            )
            st.plotly_chart(fig_returns)
            
            # Sharpe ratios
            st.subheader('⚡ Risk-Return Analysis')
            
            sharpe_ratios = {}
            for stock in returns.columns:
                sharpe_ratios[stock] = calculate_sharpe_ratio(returns[stock])
            
            sharpe_df = pd.DataFrame({
                'Stock': [STOCKS[stock]['name'] for stock in sharpe_ratios.keys()],
                'Symbol': list(sharpe_ratios.keys()),
                'Sharpe Ratio': list(sharpe_ratios.values())
            }).sort_values('Sharpe Ratio', ascending=False)
            
            fig_sharpe = px.bar(
                sharpe_df,
                x='Symbol',
                y='Sharpe Ratio',
                title='Sharpe Ratios by Stock',
                color='Sharpe Ratio',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig_sharpe)
            
            # Summary statistics
            st.subheader('📋 Summary Statistics')
            
            summary_stats = pd.DataFrame({
                'Stock': [STOCKS[stock]['name'] for stock in returns.columns],
                'Mean Daily Return (%)': (returns.mean() * 100).round(4),
                'Volatility (%)': (returns.std() * 100).round(4),
                'Sharpe Ratio': [sharpe_ratios[stock] for stock in returns.columns]
            }).round(4)
            
            st.dataframe(summary_stats, use_container_width=True)

def main():
    st.set_page_config(
        page_title='Portfolio Analysis',
        page_icon='📈',
        layout='wide'
    )
    
    # Sidebar navigation
    st.sidebar.title('🧭 Navigation')
    page = st.sidebar.radio(
        'Select Page:',
        ['Homepage', 'Portfolio Optimization', 'Analysis']
    )
    
    # Route to pages
    if page == 'Homepage':
        page_home()
    elif page == 'Portfolio Optimization':
        page_optimization()
    elif page == 'Analysis':
        page_analysis()

if __name__ == '__main__':
    main()
