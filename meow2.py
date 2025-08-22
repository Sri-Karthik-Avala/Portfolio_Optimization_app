import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import warnings

# Set option to suppress matplotlib warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore')

# Define stocks and initial investment
stocks = ['HCLTECH.NS', 'ADANIENT.NS', 'TECHM.NS', 'INFY.NS', 'WIPRO.NS', 
          'OFSS.NS', 'MPHASIS.NS', 'LTIM.NS', 'PERSISTENT.NS', 'TCS.NS']
initial_investment = 50000  # 50k INR initial investment per stock

# Alternative stock symbols to try if main ones fail
alternative_stocks = {
    'HCLTECH.NS': ['HCLTECH.BO', 'HCL-TECH.NS'],
    'ADANIENT.NS': ['ADANIENT.BO', 'ADANIENT.NSE'],
    'TECHM.NS': ['TECHM.BO', 'TECH-M.NS'],
    'INFY.NS': ['INFY.BO', 'INFY'],
    'WIPRO.NS': ['WIPRO.BO', 'WIPRO'],
    'OFSS.NS': ['OFSS.BO'],
    'MPHASIS.NS': ['MPHASIS.BO'],
    'LTIM.NS': ['LTIM.BO', 'LTI.NS'],
    'PERSISTENT.NS': ['PERSISTENT.BO'],
    'TCS.NS': ['TCS.BO']
}

# Fetch historical data function with improved error handling and retries
def fetch_data(stocks, start_date, end_date):
    import time
    import random
    
    data = pd.DataFrame()
    failed_stocks = []
    successful_stocks = []
    
    st.info(f"Fetching data for {len(stocks)} stocks from {start_date} to {end_date}...")
    progress_bar = st.progress(0)
    
    for i, stock in enumerate(stocks):
        try:
            # Add small random delay to avoid rate limiting
            time.sleep(random.uniform(0.1, 0.3))
            
            # Try multiple methods to fetch data
            stock_data = None
            
            # Method 1: Standard download with different parameters
            try:
                stock_data = yf.download(stock, start=start_date, end=end_date, 
                                       progress=False, auto_adjust=True, prepost=False, threads=True)
            except:
                pass
            
            # Method 2: Use Ticker object
            if stock_data is None or stock_data.empty:
                try:
                    ticker = yf.Ticker(stock)
                    stock_data = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                except:
                    pass
            
            # Method 3: Try with period instead of dates
            if stock_data is None or stock_data.empty:
                try:
                    ticker = yf.Ticker(stock)
                    stock_data = ticker.history(period="1y", auto_adjust=True)
                except:
                    pass
            
            # Check if we got valid data
            if stock_data is not None and not stock_data.empty:
                # Try different price columns
                price_col = None
                if 'Open' in stock_data.columns:
                    price_col = 'Open'
                elif 'Close' in stock_data.columns:
                    price_col = 'Close'
                elif 'Adj Close' in stock_data.columns:
                    price_col = 'Adj Close'
                
                if price_col is not None and not stock_data[price_col].empty:
                    # Filter data to requested date range if needed
                    stock_data = stock_data.loc[start_date:end_date]
                    if not stock_data.empty:
                        data[stock] = stock_data[price_col]
                        successful_stocks.append(stock)
                        st.success(f"✅ Successfully fetched data for {stock}")
                    else:
                        failed_stocks.append(stock)
                        st.warning(f"⚠️ No data in date range for {stock}")
                else:
                    failed_stocks.append(stock)
                    st.warning(f"⚠️ No price data available for {stock}")
            else:
                failed_stocks.append(stock)
                st.warning(f"❌ Failed to fetch any data for {stock}")
                
        except Exception as e:
            failed_stocks.append(stock)
            st.error(f"❌ Error fetching {stock}: {str(e)}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(stocks))
    
    progress_bar.empty()
    
    if data.empty:
        st.error("❌ No stock data could be fetched. This could be due to:")
        st.write("- Yahoo Finance API issues (temporary)")
        st.write("- Incorrect stock symbols")
        st.write("- Network connectivity issues")
        st.write("- Date range issues (try a different date range)")
        st.write("")
        st.write("**Suggestions:**")
        st.write("1. Try again in a few minutes (API issues are often temporary)")
        st.write("2. Check if the stock symbols are correct")
        st.write("3. Try a different date range")
        st.write("4. Check your internet connection")
        return pd.DataFrame()
    
    st.success(f"✅ Successfully fetched data for {len(successful_stocks)} out of {len(stocks)} stocks")
    
    if failed_stocks:
        st.warning(f"⚠️ Could not fetch data for: {', '.join(failed_stocks)}")
    
    # Clean and process the data
    if not data.empty:
        # Remove any columns with all NaN values
        data = data.dropna(axis=1, how='all')
        
        # Forward fill missing values
        data = data.ffill()
        
        # Drop rows with any remaining NaN values
        initial_rows = len(data)
        data = data.dropna()
        final_rows = len(data)
        
        if initial_rows > final_rows:
            st.info(f"Removed {initial_rows - final_rows} rows with missing data")
    
    return data

# Calculate portfolio value function with validation
def calculate_portfolio_value(initial_investment, data):
    if data.empty or len(data) == 0:
        st.error("Cannot calculate portfolio value: No data available")
        return pd.DataFrame(), pd.Series()
    
    # Calculate weights based on initial prices
    first_row = data.iloc[0]
    weights = initial_investment / first_row
    
    # Calculate portfolio value for each day
    portfolio_values = []
    for idx, row in data.iterrows():
        portfolio_value = np.sum(weights * row)
        portfolio_values.append(portfolio_value)
    
    data_copy = data.copy()
    data_copy['portfolio_value'] = portfolio_values
    
    return data_copy, weights

# Calculate daily returns function with validation
def calculate_daily_returns(data):
    if data.empty:
        return pd.DataFrame()
    return data.pct_change(1).dropna()

# Calculate Sharpe ratio function with validation
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    if returns.empty or returns.std() == 0:
        return 0
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
    return sharpe_ratio

# Calculate daily Sharpe ratios function with validation
def calculate_daily_sharpe_ratios(returns):
    if returns.empty:
        return pd.DataFrame()
    
    sharpe_ratios = pd.DataFrame(index=returns.index, columns=returns.columns)
    
    for i in range(30, len(returns)):  # Start from 30 days to have meaningful Sharpe ratios
        daily_returns = returns.iloc[:i]
        for col in returns.columns:
            if col in daily_returns.columns and not daily_returns[col].empty:
                sharpe_ratios.iloc[i, sharpe_ratios.columns.get_loc(col)] = calculate_sharpe_ratio(daily_returns[col])
    
    return sharpe_ratios.dropna(how='all')

# Plot daily differences in Sharpe ratios function
def plot_daily_sharpe_differences(returns):
    if returns.empty:
        fig = px.line(title='No data available for Sharpe Ratio differences')
        return fig
    
    try:
        # Calculate Sharpe ratio for the entire period for each stock
        sharpe_ratios = {}
        for col in returns.columns:
            if not returns[col].empty:
                sharpe_ratios[col] = calculate_sharpe_ratio(returns[col])
        
        if not sharpe_ratios:
            fig = px.line(title='No valid Sharpe ratios calculated')
            return fig
        
        # Create a simple bar chart of Sharpe ratios
        fig = px.bar(x=list(sharpe_ratios.keys()), y=list(sharpe_ratios.values()), 
                     title='Sharpe Ratios by Stock')
        fig.update_layout(xaxis_title='Stock', yaxis_title='Sharpe Ratio')
        return fig
    except Exception as e:
        fig = px.line(title=f'Error calculating Sharpe ratios: {str(e)}')
        return fig

# Plot portfolio weights over time function
def plot_portfolio_weights(data):
    if data.empty:
        fig = px.line(title='No data available for portfolio weights')
        return fig
    
    # Get stock columns only (exclude portfolio_value)
    stock_cols = [col for col in data.columns if col != 'portfolio_value']
    if not stock_cols:
        fig = px.line(title='No stock data available')
        return fig
    
    fig = px.line(data, x=data.index, y=stock_cols, 
                  title='Stock Prices Over Time', 
                  labels={'index': 'Date', 'value': 'Price (INR)'})
    return fig

# Plot daily weight changes in percentage function
def plot_daily_weight_changes(data):
    if data.empty:
        fig = px.line(title='No data available for daily changes')
        return fig
    
    stock_cols = [col for col in data.columns if col != 'portfolio_value']
    if not stock_cols:
        fig = px.line(title='No stock data available')
        return fig
    
    daily_returns = data[stock_cols].pct_change().dropna()
    if daily_returns.empty:
        fig = px.line(title='No return data available')
        return fig
    
    fig = px.line(daily_returns, x=daily_returns.index, y=stock_cols, 
                  title='Daily Returns (%)', 
                  labels={'index': 'Date', 'value': 'Daily Return (%)'})
    return fig

# Optimize portfolio function with validation
def optimize_portfolio(returns):
    if returns.empty or len(returns.columns) == 0:
        st.error("Cannot optimize portfolio: No return data available")
        return np.array([])
    
    # Remove any columns with all NaN values
    returns_clean = returns.dropna(axis=1, how='all')
    
    if returns_clean.empty:
        st.error("Cannot optimize portfolio: No valid return data")
        return np.array([])
    
    # Number of assets
    num_assets = len(returns_clean.columns)
    
    # Mean returns and covariance matrix
    mean_returns = returns_clean.mean()
    cov_matrix = returns_clean.cov()
    
    # Check if covariance matrix is valid
    if cov_matrix.isnull().any().any():
        st.warning("Covariance matrix contains NaN values. Using equal weights.")
        return np.array([1.0 / num_assets] * num_assets)
    
    # Risk-free rate
    risk_free_rate = 0
    
    # Objective function: negative Sharpe ratio
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        if portfolio_variance <= 0:
            return 1e10  # Return a large positive number for invalid portfolios
        portfolio_std = np.sqrt(portfolio_variance)
        if portfolio_std == 0:
            return 1e10
        return -(portfolio_return - risk_free_rate) / portfolio_std

    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess (equal distribution)
    init_guess = np.array([1.0 / num_assets] * num_assets)
    
    try:
        # Optimize
        opt_result = minimize(neg_sharpe_ratio, init_guess, 
                             args=(mean_returns, cov_matrix, risk_free_rate), 
                             method='SLSQP', bounds=bounds, constraints=constraints)
        
        if opt_result.success:
            return opt_result.x
        else:
            st.warning("Optimization failed. Using equal weights.")
            return init_guess
    except Exception as e:
        st.warning(f"Optimization error: {str(e)}. Using equal weights.")
        return init_guess

# Page 1: Homepage
def page_home():
    st.title('Welcome to Portfolio Analysis')
    try:
        st.image('page1.png', use_column_width=True)
    except:
        st.info("Homepage image not found")
    st.write('This is the homepage of the Portfolio Analysis app.')
    st.write('Use the sidebar to navigate to different sections:')
    st.write('- **Portfolio Optimization**: Find optimal portfolio weights')
    st.write('- **Analysis**: View detailed portfolio analysis and charts')

# Page 2: Portfolio Optimization
def page_portfolio_optimization():
    st.title('Portfolio Optimization')
    try:
        st.image('page2.png', use_column_width=True)
    except:
        st.info("Page 2 image not found")
    
    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Start Date', value=pd.to_datetime('2024-01-01'))
    with col2:
        end_date = st.date_input('End Date', value=pd.Timestamp.today())
    
    # Add retry button and tips
    col1, col2 = st.columns([1, 2])
    with col1:
        run_button = st.button('🚀 Run Optimization', type="primary")
    with col2:
        retry_button = st.button('🔄 Retry with Different Settings')
    
    # Tips section
    with st.expander("💡 Tips for Data Fetching Issues"):
        st.write("**If you're getting 'No stock data could be fetched' error:**")
        st.write("1. **Yahoo Finance API**: Sometimes has temporary issues - wait a few minutes and retry")
        st.write("2. **Stock Symbols**: Make sure Indian stocks end with .NS (e.g., TCS.NS)")
        st.write("3. **Date Range**: Try a different date range (avoid weekends/holidays)")
        st.write("4. **Network**: Check your internet connection")
        st.write("5. **Rate Limiting**: If multiple requests fail, wait 1-2 minutes before retrying")
    
    if run_button or retry_button:
        with st.spinner('Fetching data and optimizing portfolio...'):
            # Fetch historical data
            basedata = fetch_data(stocks, start_date.strftime('%Y-%m-%d'), 
                                end_date.strftime('%Y-%m-%d'))
            
            if basedata.empty:
                st.error("No data available for analysis. Please check the date range and stock symbols.")
                return
            
            # Calculate daily returns
            returns = calculate_daily_returns(basedata)
            
            if returns.empty:
                st.error("Could not calculate returns from the data.")
                return
            
            # Optimize portfolio
            opt_weights = optimize_portfolio(returns)
            
            if len(opt_weights) == 0:
                st.error("Portfolio optimization failed.")
                return
            
            # Display optimal weights
            st.subheader('Optimal Portfolio Weights')
            available_stocks = [stock for stock in stocks if stock in basedata.columns]
            opt_weights_df = pd.DataFrame(opt_weights, index=available_stocks, columns=['Weight'])
            opt_weights_df['Weight %'] = opt_weights_df['Weight'] * 100
            st.dataframe(opt_weights_df)
            
            # Calculate portfolio values and weights
            basedata_with_portfolio, weights = calculate_portfolio_value(initial_investment, basedata)
            
            if not basedata_with_portfolio.empty:
                # Plot portfolio value over time
                fig_portfolio_value = px.line(basedata_with_portfolio, x=basedata_with_portfolio.index, 
                                            y='portfolio_value', title='Portfolio Value Over Time')
                fig_portfolio_value.update_layout(xaxis_title='Date', yaxis_title='Portfolio Value (INR)')
                st.plotly_chart(fig_portfolio_value)

                # Display summary statistics
                st.subheader('Portfolio Performance Summary')
                initial_value = basedata_with_portfolio['portfolio_value'].iloc[0]
                final_value = basedata_with_portfolio['portfolio_value'].iloc[-1]
                total_return = (final_value - initial_value) / initial_value * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Initial Value', f'₹{initial_value:,.0f}')
                with col2:
                    st.metric('Final Value', f'₹{final_value:,.0f}')
                with col3:
                    st.metric('Total Return', f'{total_return:.2f}%')

# Page 3: Analysis
def page_analysis():
    st.title('Analysis')
    try:
        st.image('page3.png', use_column_width=True)
    except:
        st.info("Page 3 image not found")
    
    # Date selection
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Analysis Start Date', value=pd.to_datetime('2024-01-01'))
    with col2:
        end_date = st.date_input('Analysis End Date', value=pd.Timestamp.today())
    
    if st.button('Run Analysis'):
        with st.spinner('Fetching data and running analysis...'):
            # Fetch historical data
            basedata = fetch_data(stocks, start_date.strftime('%Y-%m-%d'), 
                                end_date.strftime('%Y-%m-%d'))
            
            if basedata.empty:
                st.error("No data available for analysis.")
                return
            
            # Calculate daily returns
            returns = calculate_daily_returns(basedata)
            
            # Calculate portfolio values and weights
            basedata_with_portfolio, weights = calculate_portfolio_value(initial_investment, basedata)
            
            if not basedata_with_portfolio.empty:
                # Plot Sharpe ratios
                st.subheader('Sharpe Ratios Analysis')
                fig_sharpe = plot_daily_sharpe_differences(returns)
                st.plotly_chart(fig_sharpe)

                # Plot stock prices over time
                st.subheader('Stock Prices Over Time')
                fig_prices = plot_portfolio_weights(basedata)
                st.plotly_chart(fig_prices)

                # Plot daily returns
                st.subheader('Daily Returns')
                fig_returns = plot_daily_weight_changes(basedata)
                st.plotly_chart(fig_returns)

                # Display initial weights
                st.subheader('Initial Stock Allocation')
                if not weights.empty:
                    weights_df = pd.DataFrame({
                        'Stock': weights.index,
                        'Shares': weights.values,
                        'Initial Price': basedata.iloc[0][weights.index].values,
                        'Investment': weights.values * basedata.iloc[0][weights.index].values
                    })
                    weights_df['Investment %'] = weights_df['Investment'] / weights_df['Investment'].sum() * 100
                    st.dataframe(weights_df)

                # Display summary statistics
                st.subheader('Summary Statistics')
                if not returns.empty:
                    summary_stats = pd.DataFrame({
                        'Mean Daily Return (%)': returns.mean() * 100,
                        'Volatility (%)': returns.std() * 100,
                        'Sharpe Ratio': returns.apply(calculate_sharpe_ratio)
                    })
                    st.dataframe(summary_stats)

# Main function to control page navigation
def main():
    pages = {
        "Homepage": page_home,
        "Portfolio Optimization": page_portfolio_optimization,
        "Analysis": page_analysis,
    }

    st.set_page_config(page_title='Portfolio Analysis App', layout='wide', 
                       page_icon='📈', initial_sidebar_state='auto')

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Display current selection info
    st.sidebar.write(f"Current page: {selection}")
    
    page = pages[selection]
    page()

if __name__ == '__main__':
    main()
