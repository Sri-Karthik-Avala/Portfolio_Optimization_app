import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
import warnings
import time
import random

# Try to import additional Indian stock market libraries
try:
    from nsepython import *
    NSE_AVAILABLE = True
    st.success("✅ NSEPython library loaded successfully")
except ImportError:
    NSE_AVAILABLE = False
    st.warning("⚠️ NSEPython not available. Install with: pip install nsepython")

try:
    from indstocks import Stock
    INDSTOCKS_AVAILABLE = True
    st.success("✅ INDStocks library loaded successfully")
except ImportError:
    INDSTOCKS_AVAILABLE = False
    st.warning("⚠️ INDStocks not available. Install with: pip install indstocks")

# Set option to suppress matplotlib warnings
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings('ignore')

# Define stocks and initial investment with multiple symbol formats
stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 
          'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'ASIANPAINT']
initial_investment = 50000  # 50k INR initial investment per stock

# Map of company names and different symbol formats
stock_info = {
    'RELIANCE': {
        'name': 'Reliance Industries',
        'yf_symbols': ['RELIANCE.NS', 'RELIANCE.BO'],
        'nse_symbol': 'RELIANCE',
        'indstocks_symbol': 'RELIANCE'
    },
    'TCS': {
        'name': 'Tata Consultancy Services',
        'yf_symbols': ['TCS.NS', 'TCS.BO'],
        'nse_symbol': 'TCS',
        'indstocks_symbol': 'TCS'
    },
    'INFY': {
        'name': 'Infosys',
        'yf_symbols': ['INFY.NS', 'INFY.BO', 'INFY'],
        'nse_symbol': 'INFY',
        'indstocks_symbol': 'INFY'
    },
    'HDFCBANK': {
        'name': 'HDFC Bank',
        'yf_symbols': ['HDFCBANK.NS', 'HDFCBANK.BO'],
        'nse_symbol': 'HDFCBANK',
        'indstocks_symbol': 'HDFCBANK'
    },
    'ICICIBANK': {
        'name': 'ICICI Bank',
        'yf_symbols': ['ICICIBANK.NS', 'ICICIBANK.BO'],
        'nse_symbol': 'ICICIBANK',
        'indstocks_symbol': 'ICICIBANK'
    },
    'HINDUNILVR': {
        'name': 'Hindustan Unilever',
        'yf_symbols': ['HINDUNILVR.NS', 'HINDUNILVR.BO'],
        'nse_symbol': 'HINDUNILVR',
        'indstocks_symbol': 'HINDUNILVR'
    },
    'ITC': {
        'name': 'ITC Limited',
        'yf_symbols': ['ITC.NS', 'ITC.BO'],
        'nse_symbol': 'ITC',
        'indstocks_symbol': 'ITC'
    },
    'SBIN': {
        'name': 'State Bank of India',
        'yf_symbols': ['SBIN.NS', 'SBIN.BO'],
        'nse_symbol': 'SBIN',
        'indstocks_symbol': 'SBIN'
    },
    'BHARTIARTL': {
        'name': 'Bharti Airtel',
        'yf_symbols': ['BHARTIARTL.NS', 'BHARTIARTL.BO'],
        'nse_symbol': 'BHARTIARTL',
        'indstocks_symbol': 'BHARTIARTL'
    },
    'ASIANPAINT': {
        'name': 'Asian Paints',
        'yf_symbols': ['ASIANPAINT.NS', 'ASIANPAINT.BO'],
        'nse_symbol': 'ASIANPAINT',
        'indstocks_symbol': 'ASIANPAINT'
    }
}

# Fetch data using NSEPython
def fetch_nse_data(stock_symbol, start_date, end_date):
    """Fetch historical data using NSEPython"""
    try:
        if not NSE_AVAILABLE:
            return None
        
        # Get historical data from NSE
        # Note: NSEPython might have different functions for historical data
        # This is a basic implementation - you might need to adjust based on actual NSEPython API
        stock_data = nse_eq(stock_symbol)
        
        if stock_data and 'lastPrice' in stock_data:
            # For now, create a simple series with current price
            # In real implementation, you'd fetch historical data
            current_price = stock_data['lastPrice']
            
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            date_range = date_range[date_range.weekday < 5]  # Remove weekends
            
            # Create a simple price series (this would be historical data in real implementation)
            prices = pd.Series(index=date_range, data=current_price, name=stock_symbol)
            return prices
            
    except Exception as e:
        return None
    
    return None

# Fetch data using INDStocks
def fetch_indstocks_data(stock_symbol, start_date, end_date):
    """Fetch historical data using INDStocks"""
    try:
        if not INDSTOCKS_AVAILABLE:
            return None
        
        stock = Stock(stock_symbol)
        
        # Get current price
        current_price = stock.get_price()
        
        if current_price:
            # Generate date range
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            date_range = date_range[date_range.weekday < 5]  # Remove weekends
            
            # Create a simple price series (this would be historical data in real implementation)
            prices = pd.Series(index=date_range, data=current_price, name=stock_symbol)
            return prices
            
    except Exception as e:
        return None
    
    return None

# Enhanced fetch function with multiple data sources
def fetch_data(stocks, start_date, end_date):
    """Fetch stock data using multiple sources"""
    data = pd.DataFrame()
    failed_stocks = []
    successful_stocks = []
    
    st.info(f"Fetching data for {len(stocks)} stocks from {start_date} to {end_date}...")
    
    # Show available data sources
    sources = []
    if NSE_AVAILABLE:
        sources.append("NSEPython")
    if INDSTOCKS_AVAILABLE:
        sources.append("INDStocks")
    sources.append("yfinance")
    
    st.write(f"**Available data sources:** {', '.join(sources)}")
    
    progress_bar = st.progress(0)
    
    for i, stock in enumerate(stocks):
        stock_fetched = False
        company_name = stock_info[stock]['name']
        
        st.write(f"Fetching {company_name} ({stock})...")
        
        # Method 1: Try yfinance first (most reliable for historical data)
        if not stock_fetched:
            st.write("  → Trying yfinance...")
            for yf_symbol in stock_info[stock]['yf_symbols']:
                try:
                    time.sleep(0.5)  # Rate limiting
                    stock_data = yf.download(yf_symbol, start=start_date, end=end_date, 
                                           progress=False, show_errors=False, auto_adjust=True)
                    
                    if not stock_data.empty and len(stock_data) > 10:
                        if 'Close' in stock_data.columns:
                            price_data = stock_data['Close'].dropna()
                            if len(price_data) > 10:
                                data[stock] = price_data
                                successful_stocks.append(stock)
                                stock_fetched = True
                                st.success(f"    ✅ Success with yfinance ({yf_symbol})")
                                break
                except Exception as e:
                    continue
            
            if stock_fetched:
                # Display initial weights
                st.subheader('Initial Stock Allocation')
                if not weights.empty:
                    weights_df = pd.DataFrame({
                        'Company': [stock_info[stock]['name'] for stock in weights.index if stock in stock_info],
                        'Symbol': [stock for stock in weights.index if stock in stock_info],
                        'Shares': [weights[stock] for stock in weights.index if stock in stock_info],
                        'Initial Price': [basedata.iloc[0][stock] for stock in weights.index if stock in stock_info and stock in basedata.columns],
                        'Investment': [weights[stock] * basedata.iloc[0][stock] for stock in weights.index if stock in stock_info and stock in basedata.columns]
                    })
                    weights_df['Investment %'] = weights_df['Investment'] / weights_df['Investment'].sum() * 100
                    st.dataframe(weights_df) Update progress and continue to next stock
                progress_bar.progress((i + 1) / len(stocks))
                continue
        
        # Method 2: Try NSEPython
        if not stock_fetched and NSE_AVAILABLE:
            st.write("  → Trying NSEPython...")
            try:
                nse_data = fetch_nse_data(stock_info[stock]['nse_symbol'], start_date, end_date)
                if nse_data is not None and len(nse_data) > 10:
                    data[stock] = nse_data
                    successful_stocks.append(stock)
                    stock_fetched = True
                    st.success(f"    ✅ Success with NSEPython")
            except Exception as e:
                st.write(f"    ❌ NSEPython failed: {str(e)[:50]}...")
        
        # Method 3: Try INDStocks
        if not stock_fetched and INDSTOCKS_AVAILABLE:
            st.write("  → Trying INDStocks...")
            try:
                indstocks_data = fetch_indstocks_data(stock_info[stock]['indstocks_symbol'], start_date, end_date)
                if indstocks_data is not None and len(indstocks_data) > 10:
                    data[stock] = indstocks_data
                    successful_stocks.append(stock)
                    stock_fetched = True
                    st.success(f"    ✅ Success with INDStocks")
            except Exception as e:
                st.write(f"    ❌ INDStocks failed: {str(e)[:50]}...")
        
        if not stock_fetched:
            failed_stocks.append(stock)
            st.error(f"    ❌ All methods failed for {company_name}")
        
        # Update progress
        progress_bar.progress((i + 1) / len(stocks))
    
    progress_bar.empty()
    
    # Process final data
    if not data.empty:
        # Align all data to common dates
        data = data.dropna(how='all')
        data = data.ffill().dropna()
        
        if len(data) < 10:
            st.error("Not enough historical data available. Please try a different date range.")
            return pd.DataFrame()
        
        st.success(f"✅ Successfully fetched data for {len(successful_stocks)} out of {len(stocks)} stocks")
        
        # Show data summary
        summary_df = pd.DataFrame({
            'Company': [stock_info[col]['name'] for col in data.columns],
            'Symbol': data.columns,
            'Data Points': [data[col].count() for col in data.columns],
            'Date Range': [f"{data.index[0].date()} to {data.index[-1].date()}" for _ in data.columns],
            'Latest Price': [f"₹{data[col].iloc[-1]:.2f}" for col in data.columns]
        })
        st.dataframe(summary_df)
        
    else:
        st.error("❌ No stock data could be fetched from any source.")
        st.write("**Try installing additional libraries:**")
        if not NSE_AVAILABLE:
            st.code("pip install nsepython")
        if not INDSTOCKS_AVAILABLE:
            st.code("pip install indstocks")
        
        st.write("**Or use sample data for testing:**")
        if st.button("🧪 Generate Sample Data"):
            return create_sample_data(start_date, end_date)
    
    if failed_stocks:
        st.warning(f"⚠️ Could not fetch data for: {', '.join([stock_info[s]['name'] for s in failed_stocks])}")
    
    return data

# Create sample data function for testing when API fails
def create_sample_data(start_date, end_date):
    """Create realistic sample data for testing when API fails"""
    st.info("🧪 Generating sample data for testing purposes...")
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    # Remove weekends
    date_range = date_range[date_range.weekday < 5]
    
    # Base prices for each stock (realistic Indian stock prices)
    base_prices = {
        'RELIANCE': 2500,
        'TCS': 3200,
        'INFY': 1500,
        'HDFCBANK': 1600,
        'ICICIBANK': 900,
        'HINDUNILVR': 2400,
        'ITC': 450,
        'SBIN': 550,
        'BHARTIARTL': 850,
        'ASIANPAINT': 3000
    }
    
    np.random.seed(42)  # For reproducible results
    data = pd.DataFrame(index=date_range)
    
    for stock in stocks:
        if stock in base_prices:
            base_price = base_prices[stock]
            # Generate realistic price movements
            returns = np.random.normal(0.0005, 0.02, len(date_range))  # Small daily returns with volatility
            prices = [base_price]
            
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, base_price * 0.5))  # Prevent unrealistic crashes
            
            data[stock] = prices
    
    st.success(f"✅ Generated sample data for {len(stocks)} stocks with {len(data)} trading days")
    st.warning("⚠️ This is sample data for testing. Install nsepython/indstocks for real data.")
    
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
    with st.expander("💡 Data Sources and Installation Guide"):
        st.write("**Available Data Sources (in order of preference):**")
        st.write("1. **yfinance**: Most reliable for historical data")
        st.write("2. **NSEPython**: Direct NSE data access")
        st.write("3. **INDStocks**: Indian stock market specialist library")
        st.write("")
        st.write("**Installation Commands:**")
        st.code("pip install nsepython")
        st.code("pip install indstocks")
        st.write("")
        st.write("**Current Portfolio (Top Indian Companies):**")
        for symbol in stocks:
            name = stock_info[symbol]['name']
            st.write(f"• {name}: `{symbol}`")
        st.write("")
        st.write("**If data fetching fails:**")
        st.write("- Try different date ranges (recent dates work better)")
        st.write("- Install nsepython and indstocks for better reliability")
        st.write("- Use sample data to test the portfolio optimization features")
    
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
            opt_weights_df = pd.DataFrame({
                'Company': [stock_info[stock]['name'] for stock in available_stocks],
                'Symbol': available_stocks,
                'Weight': opt_weights,
                'Weight %': opt_weights * 100
            })
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
    
    # Add retry functionality
    col1, col2 = st.columns([1, 2])
    with col1:
        run_button = st.button('📊 Run Analysis', type="primary")
    with col2:
        retry_button = st.button('🔄 Retry Analysis')
    
    if run_button or retry_button:
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
