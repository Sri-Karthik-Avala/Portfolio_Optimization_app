import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set option to suppress matplotlib warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define stocks
stocks = ['HCLTECH.NS', 'ADANIENT.NS', 'TECHM.NS', 'INFY', 'WIPRO.NS', 
          'OFSS.NS', 'MPHASIS.NS', 'LTIM.NS', 'PERSISTENT.NS', 'TCS.NS']

# Define initial investment per stock (0.1 of the overall budget)
initial_budget = 500000
initial_investment = 50000  # 50k INR initial investment per stock

# Function to fetch historical data for stocks
def fetch_data(stocks, start_date, end_date):
    data = pd.DataFrame()
    for stock in stocks:
        stock_data = yf.download(stock, start=start_date, end=end_date)
        data[stock] = stock_data['Open']
    return data

# Function to calculate portfolio value
def PortfolioCalc(initial_investment, data):
    # Calculate weights based on initial investment
    weights = initial_investment / data.iloc[0]  # Divide initial investment by first day's prices
    data['portfolio_value'] = np.sum(weights * data, axis=1)
    return data, weights

# Function to calculate daily returns
def calculate_daily_returns(data):
    return data.pct_change(1)

# Function to calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0):
    sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
    return sharpe_ratio

# Function to calculate daily Sharpe ratios
def calculate_daily_sharpe_ratios(returns):
    sharpe_ratios = pd.DataFrame(index=returns.index, columns=returns.columns)
    for i in range(1, len(returns)):
        daily_returns = returns.iloc[:i]
        sharpe_ratios.iloc[i] = daily_returns.apply(calculate_sharpe_ratio, axis=0)
    return sharpe_ratios

# Function to plot daily differences in Sharpe ratios
def plot_daily_sharpe_differences(returns):
    daily_sharpe = returns.apply(calculate_sharpe_ratio, axis=1)
    daily_sharpe_diff = daily_sharpe.diff()
    plt.figure(figsize=(10, 6))
    plt.plot(daily_sharpe_diff.index, daily_sharpe_diff.values, marker='o', linestyle='-')
    plt.title('Daily Differences in Sharpe Ratios')
    plt.xlabel('Date')
    plt.ylabel('Difference in Sharpe Ratio')
    plt.grid(True)
    st.pyplot()

# Function to plot portfolio weights over time
def plot_portfolio_weights(data):
    plt.figure(figsize=(10, 6))
    for stock in stocks:
        plt.plot(data.index, data[stock], label=stock)
    plt.title('Portfolio Weights Over Time')
    plt.xlabel('Date')
    plt.ylabel('Weight')
    plt.legend(loc='upper left')
    plt.grid(True)
    st.pyplot()

# Function to plot daily weight changes in percentage
def plot_daily_weight_changes(data):
    daily_returns = data.pct_change()
    plt.figure(figsize=(10, 6))
    for stock in stocks:
        plt.plot(daily_returns.index, daily_returns[stock] * 100, label=stock)
    plt.title('Daily Weight Changes (%)')
    plt.xlabel('Date')
    plt.ylabel('Daily Weight Change (%)')
    plt.legend(loc='upper left')
    plt.grid(True)
    st.pyplot()

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
    
    # Display selected stocks
    st.subheader('Selected Stocks')
    st.write(stocks)
    
    # Fetch historical data
    basedata = fetch_data(stocks, '2024-06-01', end_date)  # Adjust start date as needed
    
    # Forward fill any missing data
    basedata = basedata.ffill()
    
    # Calculate portfolio values and weights
    basedata, weights = PortfolioCalc(initial_investment, basedata)
    
    # Convert weights to DataFrame
    weights_df = pd.DataFrame(weights.values, index=weights.index, columns=['Weight'])
    
    # Plot portfolio value
    st.subheader('Portfolio Value Over Time')
    plt.figure(figsize=(10, 6))
    plt.plot(basedata.index, basedata['portfolio_value'], label='Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.legend()
    st.pyplot()

    # Display table of data
    st.subheader('Portfolio Data')
    st.write(basedata)

    # Display weights table
    st.subheader('First Change Stock Quantities Table')
    st.write(weights_df)

# Updated Analysis page function
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
    basedata, weights = PortfolioCalc(initial_investment, basedata)
    
    # Record daily changes in allocation
    daily_changes = basedata[stocks].diff() * weights.values[0]
    daily_change_in_allocation = pd.DataFrame(daily_changes, columns=stocks)
    
    # Calculate daily Sharpe ratios for each stock
    daily_sharpe_ratios_df = calculate_daily_sharpe_ratios(returns)
    
    # Plot daily differences in Sharpe ratios
    st.subheader('Daily Differences in Sharpe Ratios')
    plot_daily_sharpe_differences(returns)
    
    # Plot portfolio weights over time
    st.subheader('Portfolio Weights Over Time')
    plot_portfolio_weights(basedata)
    
    # Plot daily weight changes in percentage
    st.subheader('Daily Weight Changes (%)')
    plot_daily_weight_changes(basedata)
    
    # Display weights table
    st.subheader('First Change Stock Quantities Table')
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
    st.subheader('Extra Values')
    st.write('Returns Data:')
    st.write(returns)
    st.write('Change in Stock Price Data:')
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
