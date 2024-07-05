import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize

# Set option to suppress matplotlib warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define stocks and initial investment
stocks = ['HCLTECH.NS', 'ADANIENT.NS', 'TECHM.NS', 'INFY', 'WIPRO.NS', 
          'OFSS.NS', 'MPHASIS.NS', 'LTIM.NS', 'PERSISTENT.NS', 'TCS.NS']
initial_investment = 50000  # 50k INR initial investment per stock

# Fetch historical data function
def fetch_data(stocks, start_date, end_date):
    data = pd.DataFrame()
    for stock in stocks:
        stock_data = yf.download(stock, start=start_date, end=end_date)
        data[stock] = stock_data['Open']
    return data

# Calculate portfolio value function
def calculate_portfolio_value(initial_investment, data):
    weights = initial_investment / data.iloc[0]
    data['portfolio_value'] = np.sum(weights * data, axis=1)
    return data, weights

# Calculate daily returns function
def calculate_daily_returns(data):
    return data.pct_change(1)

# Calculate Sharpe ratio function
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
    return sharpe_ratio

# Calculate daily Sharpe ratios function
def calculate_daily_sharpe_ratios(returns):
    sharpe_ratios = pd.DataFrame(index=returns.index, columns=returns.columns)
    for i in range(1, len(returns)):
        daily_returns = returns.iloc[:i]
        sharpe_ratios.iloc[i] = daily_returns.apply(calculate_sharpe_ratio, axis=0)
    return sharpe_ratios

# Plot daily differences in Sharpe ratios function
def plot_daily_sharpe_differences(returns):
    daily_sharpe = returns.apply(calculate_sharpe_ratio, axis=1)
    daily_sharpe_diff = daily_sharpe.diff()
    fig = px.line(x=daily_sharpe_diff.index, y=daily_sharpe_diff.values, title='Daily Differences in Sharpe Ratios')
    return fig

# Plot portfolio weights over time function
def plot_portfolio_weights(data):
    fig = px.line(data, x=data.index, y=stocks, title='Portfolio Weights Over Time', labels={'index': 'Date', 'value': 'Weight'})
    return fig

# Plot daily weight changes in percentage function
def plot_daily_weight_changes(data):
    daily_returns = data.pct_change()
    fig = px.line(daily_returns, x=daily_returns.index, y=stocks, title='Daily Weight Changes (%)', labels={'index': 'Date', 'value': 'Daily Weight Change (%)'})
    return fig

# Optimize portfolio function
def optimize_portfolio(returns):
    # Number of assets
    num_assets = len(returns.columns)
    
    # Mean returns and covariance matrix
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Risk-free rate
    risk_free_rate = 0
    
    # Objective function: negative Sharpe ratio
    def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - risk_free_rate) / portfolio_std

    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Initial guess (equal distribution)
    init_guess = num_assets * [1. / num_assets]
    
    # Optimize
    opt_result = minimize(neg_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, risk_free_rate), 
                          method='SLSQP', bounds=bounds, constraints=constraints)
    
    return opt_result.x

# Page 1: Homepage
def page_home():
    st.title('Welcome to Portfolio Analysis')
    st.image('page1.png', use_column_width=True)
    st.write('This is the homepage of the Portfolio Analysis app.')

# Page 2: Portfolio Optimization
def page_portfolio_optimization():
    st.title('Portfolio Optimization')
    st.image('page2.png', use_column_width=True)
    
    # Select end date (today's date)
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # Fetch historical data
    basedata = fetch_data(stocks, '2024-06-01', end_date)  # Adjust start date as needed
    
    # Forward fill any missing data
    basedata = basedata.ffill()
    
    # Calculate daily returns
    returns = calculate_daily_returns(basedata)
    
    # Optimize portfolio
    opt_weights = optimize_portfolio(returns)
    
    # Display optimal weights
    st.subheader('Optimal Portfolio Weights')
    opt_weights_df = pd.DataFrame(opt_weights, index=stocks, columns=['Weight'])
    st.write(opt_weights_df)
    
    # Calculate portfolio values and weights
    basedata, weights = calculate_portfolio_value(initial_investment, basedata)
    
    # Plot portfolio value over time
    fig_portfolio_value = px.line(basedata, x=basedata.index, y='portfolio_value', title='Portfolio Value Over Time')
    st.plotly_chart(fig_portfolio_value)

    # Display table of data
    st.subheader('Portfolio Data')
    st.write(basedata)

# Page 3: Analysis
def page_analysis():
    st.title('Analysis')
    st.image('page3.png', use_column_width=True)
    
    # Select end date (today's date)
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    # Fetch historical data
    basedata = fetch_data(stocks, '2024-06-01', end_date)  # Adjust start date as needed
    
    # Forward fill any missing data
    basedata = basedata.ffill()
    
    # Calculate daily returns
    returns = calculate_daily_returns(basedata)
    
    # Calculate portfolio values and weights
    basedata, weights = calculate_portfolio_value(initial_investment, basedata)
    
    # Record daily changes in allocation
    daily_changes = basedata[stocks].diff() * weights.values[0]
    daily_change_in_allocation = pd.DataFrame(daily_changes, columns=stocks)
    
    # Calculate daily Sharpe ratios for each stock
    daily_sharpe_ratios_df = calculate_daily_sharpe_ratios(returns)
    
    # Plot daily differences in Sharpe ratios
    st.subheader('Daily Differences in Sharpe Ratios')
    fig_daily_sharpe_diff = plot_daily_sharpe_differences(returns)
    st.plotly_chart(fig_daily_sharpe_diff)

    # Plot portfolio weights over time
    st.subheader('Portfolio Weights Over Time')
    fig_portfolio_weights = plot_portfolio_weights(basedata)
    st.plotly_chart(fig_portfolio_weights)

    # Plot daily weight changes in percentage
    st.subheader('Daily Weight Changes (%)')
    fig_daily_weight_changes = plot_daily_weight_changes(basedata)
    st.plotly_chart(fig_daily_weight_changes)

    # Display weights table
    st.subheader('Initial Stock Weights')
    initial_weights = initial_investment / basedata.iloc[0]
    weights_df = pd.DataFrame(initial_weights.values, index=initial_weights.index, columns=['Initial Weight'])
    st.write(weights_df)
    
    # Display daily changes in allocation as a DataFrame
    st.subheader('Daily Change in Allocation')
    st.write(daily_change_in_allocation)
    
    # Display Sharpe ratios table for all companies over time
    st.subheader('Sharpe Ratios for All Companies (Daily)')
    st.write(daily_sharpe_ratios_df)
    
    # Display all tables
    st.subheader('Additional Data')
    st.write('Returns Data:')
    st.write(returns)
    st.write('Price Data:')
    st.write(basedata)

# Main function to control page navigation
def main():
    pages = {
        "Homepage": page_home,
        "Portfolio Optimization": page_portfolio_optimization,
        "Analysis": page_analysis,
    }

    st.set_page_config(page_title='Portfolio Analysis App', layout='wide', page_icon=None, initial_sidebar_state='auto')

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    page = pages[selection]
    page()

if __name__ == '__main__':
    main()
