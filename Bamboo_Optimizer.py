# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 22:36:32 2021

@author: Prince
"""

import pandas as pd
import numpy as np
import scipy.stats as scistat
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# Read csv file containing the symbols
csvFile2Load = "Bamboo_Stocks.csv"
symbolsDF = pd.read_csv(csvFile2Load, header=None, names=['S'])
nStocks = len(symbolsDF)

# Concantenate all files with prices
prices = pd.read_csv(symbolsDF['S'][0] + '.csv', header=None, names=['AAPL'])

for i in range(1, nStocks):
    stock_price = pd.read_csv(symbolsDF['S'][i] + '.csv', header=None)
    prices[symbolsDF['S'][i]] = stock_price

# Compute returns
returns = prices.pct_change()

# Compute mean daily returns and covariance matrix
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# Initialize portfolio weights: Start with equal weights: Weights must all sum to 1.
weights = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2])

# Compute portfolio return & standard deviation
portfolio_return = round(np.sum(mean_daily_returns * weights) * 252, 2)
portfolio_stdDev = round(np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252), 2)

# Plot histogram of returns: This will be used to select Target returns for Sortino Ratio Calculations
returns = returns.dropna()
ax = returns.hist(figsize=(10, 10))

# Initialize Target returns
target_returns = np.zeros(nStocks)

# Compute mode of each stock's returns
for i in range(nStocks):
    target_returns[i] = scistat.mode(returns[symbolsDF['S'][i]])[0]

# Compute minimum target return
min_targetReturn = np.mean(target_returns)

# Eliminate negative returs as a target - we are not here to loose money
if min_targetReturn < 0:
    min_targetReturn = 0

# Generally a good idea to add a buffer to the minimum target retun - The more aggressive the investor, the higher the buffer should be 
buffer_return = 0.2
target_return = min_targetReturn + buffer_return

print()    
print("Minimum Suggested Target Return", target_return)
print()

# Compute Sharpe Ratio
SharpeRatio = portfolio_return / portfolio_stdDev

# Convert returns and standard deviatio to %
portfolio_return_pct = "{:.0%}".format(portfolio_return)
portfolio_stdDev_pct = "{:.0%}".format(portfolio_stdDev)

# Output naive portfolio statistics
print("Below is the Naive Portfolio Statistics assuming equal weights")
print("Naive Expected annual return: " + portfolio_return_pct)
print("Naive anual Volatility: " + portfolio_stdDev_pct)
print("Naive Sharpe Ratio: " + str(SharpeRatio))
print()
# print("Naive Target Return: " + str(target_return))

# Expected returns and sample covariance
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)

# Optimise portfolio for maximum Sharpe Ratio
ef = EfficientFrontier(mu, S)
raw_weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("Max Sharpe Ratio weights is: ", cleaned_weights) # gives ordered dictionary of weights
print()

# Output portfolio performance based on max Sharpe Ratio
print("Below is the Maximum Sharpe Ratio Portfolio Statistics")
ef.portfolio_performance(verbose=True)
print()

# Compute number of shares to buy back to maximize Sharpe Ratio
latest_prices = get_latest_prices(prices)
da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=100000) 
allocation, leftover = da.lp_portfolio()

print("Below is the amount of each stock to buy and the funds that will remain after")
print("Discrete allocation:", allocation)
print("Funds remaining: ${:.2f}".format(leftover))
print()

# The above can be repeated for Long/Short portfolio too
ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
raw_weights = ef.efficient_return(target_return=target_return, market_neutral=True)
cleaned_weights = ef.clean_weights()

print("Below is the weights for a long/short portfolio on the efficient frontier")
print("Long/Short Ratio weights is: ", cleaned_weights) # gives ordered dictionary of weights
print()

# Output portfolio performance based on Long/Short Strategy
print("Below is the Long/Short Portfolio Statistics")
ef.portfolio_performance(verbose=True)