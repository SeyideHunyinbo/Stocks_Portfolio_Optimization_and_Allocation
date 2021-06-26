from tkinter import *
try:  # 3.X
    import tkinter as tk

except ImportError:  # 2.X
    import Tkinter as tk
    import tkk
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename

from tkcalendar import Calendar, DateEntry

import pandas as pd
import numpy as np
import scipy.stats as scistat
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


def import_csv_file():
    global csv_var
    csv_file_path = askopenfilename()
    csv_var.set(str(csv_file_path.replace("/", "\\")))
    print(csv_file_path.replace("/", "\\"))


def clear():
    portfolio_value_textbox.delete(0, "END")
    AAPL_textbox.delete(0, "END")
    AMZN_textbox.delete(0, "END")
    JNJ_textbox.delete(0, "END")
    FB_textbox.delete(0, "END")
    MSFT_textbox.delete(0, "END")
    NEAR_textbox.delete(0, "END")
    NAV_textbox.delete(0, "END")
    NSR_textbox.delete(0, "END")
    AV_textbox.delete(0, "END")
    EAR_textbox.delete(0, "END")
    SR_textbox.delete(0, "END")


def optimize_portfolio():

    # instantiating variables
    portfolioz_value = portfolio_value_textbox.get()
    AAPL_value = AAPL_textbox.get()
    AMZN_value = AMZN_textbox.get()
    JNJ_value = JNJ_textbox.get()
    FB_value = FB_textbox.get()

    # Read csv file containing symbols
    symbolsDF = pd.read_csv("C:\\Users\\hunyi\\Desktop\\StartUp-Stocks-Prince\\Bamboo_Stocks.csv",
                            header=None, names=['S'])  # hard coded to change later
    nStocks = 5

    # # Concantenate all files with prices
    # prices = pd.read_csv(symbolsDF['S'][0] +
    #                      '.csv', header=None, names=['AAPL'])

    prices = pd.read_csv(
        "C:\\Users\\hunyi\\Desktop\\StartUp-Stocks-Prince\\Stocks_All.csv", index_col=0)

    # for i in range(1, nStocks):
    #     stock_price = pd.read_csv(symbolsDF['S'][i] + '.csv', header=None)
    #     prices[symbolsDF['S'][i]] = stock_price

    # Compute returns
    returns = prices.pct_change()

    # Compute mean daily returns and covariance matrix
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    # Initialize portfolio weights: Start with equal weights: Weights must all sum to 1.
    weights = np.asarray([0.2, 0.2, 0.2, 0.2, 0.2])

    # Compute portfolio return & standard deviation
    portfolio_return = round(np.sum(mean_daily_returns * weights) * 252, 2)
    portfolio_stdDev = round(
        np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252), 2)

    # Plot histogram of returns: This will be used to select Target returns for Sortino Ratio Calculations
    returns = returns.dropna()
    #ax = returns.hist(figsize=(10, 10))

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

    # Compute Sharpe Ratio
    SharpeRatio = portfolio_return / portfolio_stdDev
    NSR_var.set(str(SharpeRatio))

    # Convert returns and standard deviatio to %
    portfolio_return_pct = "{:.0%}".format(portfolio_return)
    portfolio_stdDev_pct = "{:.0%}".format(portfolio_stdDev)

    # Expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)

    # Optimise portfolio for maximum Sharpe Ratio
    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    ef.portfolio_performance(verbose=True)
    # print()

    # Compute number of shares to buy back to maximize Sharpe Ratio
    latest_prices = get_latest_prices(prices)
    da = DiscreteAllocation(cleaned_weights, latest_prices,
                            total_portfolio_value=100000)
    allocation, leftover = da.lp_portfolio()

    #print("Below is the amount of each stock to buy and the funds that will remain after")
    #print("Discrete allocation:", allocation)
    #print("Funds remaining: ${:.2f}".format(leftover))
    # print()

    # The above can be repeated for Long/Short portfolio too
    ef = EfficientFrontier(mu, S, weight_bounds=(-1, 1))
    raw_weights = ef.efficient_return(
        target_return=target_return, market_neutral=True)
    cleaned_weights = ef.clean_weights()

    for key, value in cleaned_weights.items():
        if key == "AAPL":
            AAPL_var.set(str(value))
        if key == "AMZN":
            AMZN_var.set(str(value))
        if key == "FB":
            FB_var.set(str(value))
        if key == "JNJ":
            JNJ_var.set(str(value))
        if key == "MSFT":
            MSFT_var.set(str(value))

    #print("Below is the weights for a long/short portfolio on the efficient frontier")
    # print("Long/Short Ratio weights is: ", cleaned_weights) # gives ordered dictionary of weights
    # print()

    # Output portfolio performance based on Long/Short Strategy
    #print("Below is the Long/Short Portfolio Statistics")

    portfolio_performance = ef.portfolio_performance(verbose=True)
    EAR, AV, SRs = portfolio_performance

    SR.set(str(SRs))
    EAR_var.set(str(EAR*100)+"%")
    AV_var.set(str(AV) + "%")

    NEAR_var.set(str(portfolio_return_pct))
    NAV_var.set(str(portfolio_stdDev_pct))


root = Tk()
root.title("Portfolio Optimizer")
root.geometry("900x600")

csv_var = tk.StringVar()

frame = Frame(root, bg='white', bd='10', padx=3, pady=3)
frame.pack(side=tk.TOP, fill=tk.X)


upload_csv_Label = Label(frame, text="Upload CSV")
upload_csv_Label.grid(row=3, column=1)

button_upload_csv_file = Button(
    frame, text="Upload CSV", command=import_csv_file)
button_upload_csv_file.grid(row=3, column=2)

time_period_label = Label(frame, text='Enter Time Period :')
time_period_label.grid(row=0, column=0)

cal = Calendar(frame, selectmode="day", year=2021)
cal.grid(row=0, column=1, padx=10, pady=10)

# endtime_textbox = Entry(frame, width=35, borderwidth=5)
# endtime_textbox.grid(row=0, column=1, columnspan=2, padx=10, pady=10)

portfolio_value_label = Label(frame, text="Portfolio Value:")
portfolio_value_label.grid(row=2, column=0)


portfolio_value_textbox = Entry(frame, width=35, borderwidth=5)
portfolio_value_textbox.grid(row=2, column=1, columnspan=2, padx=10, pady=10)


button_optimize = Button(frame, text="Optimize",
                         bg="green", command=optimize_portfolio)
button_optimize.grid(row=2, column=20)


button_clear = Button(frame, text="Clear Results", bg="red", command=clear)
button_clear.grid(row=2, column=12, padx=10, pady=10)

frame1 = Frame(root, bg='white', bd='10', padx=3, pady=3)
frame1.pack(fill=tk.X, side=tk.LEFT)

ticker_label = Label(frame1, text="Ticker")
ticker_label.grid(row=1, column=1)

allocation_label = Label(frame1, text="Allocaton %")
allocation_label.grid(row=1, column=2, padx=10, pady=10)


AAPL_label = Label(frame1, text="AAPL: ")
AAPL_label.grid(row=2, column=1, padx=10, pady=10)

AAPL_var = StringVar()
AAPL_textbox = Entry(frame1, width=35, borderwidth=5, textvariable=AAPL_var)
AAPL_textbox.grid(row=2, column=2, columnspan=2, padx=10, pady=10)


AMZN_label = Label(frame1, text="AMZN: ")
AMZN_label.grid(row=3, column=1, padx=10, pady=10)

AMZN_var = StringVar()
AMZN_textbox = Entry(frame1, width=35, borderwidth=5, textvariable=AMZN_var)
AMZN_textbox.grid(row=3, column=2, columnspan=2, padx=10, pady=10)


JNJ_label = Label(frame1, text="JNJ: ")
JNJ_label.grid(row=4, column=1, padx=10, pady=10)

JNJ_var = StringVar()
JNJ_textbox = Entry(frame1, width=35, borderwidth=5, textvariable=JNJ_var)
JNJ_textbox.grid(row=4, column=2, columnspan=2, padx=10, pady=10)


FB_label = Label(frame1, text="FB ")
FB_label.grid(row=5, column=1, padx=10, pady=10)

FB_var = StringVar()
FB_textbox = Entry(frame1, width=35, borderwidth=5, textvariable=FB_var)
FB_textbox.grid(row=5, column=2, columnspan=2, padx=10, pady=10)


MSFT_label = Label(frame1, text="MSFT: ")
MSFT_label.grid(row=6, column=1, padx=10, pady=10)

MSFT_var = StringVar()
MSFT_textbox = Entry(frame1, width=35, borderwidth=5, textvariable=MSFT_var)
MSFT_textbox.grid(row=6, column=2, columnspan=2, padx=10, pady=10)


frame2 = Frame(root, bg='white', bd='10', padx=3, pady=3)
frame2.pack(fill=tk.X, side=tk.RIGHT)

NEAR_label = Label(frame2, text="Naive Expected Annual Returns: ")
NEAR_label.grid(row=1, column=1, padx=10, pady=10)
NEAR_var = StringVar()

NEAR_textbox = Entry(frame2, width=35, borderwidth=5,
                     textvariable=NEAR_var, state=DISABLED)
NEAR_textbox.grid(row=1, column=2, columnspan=2, padx=10, pady=10)

NAV_label = Label(frame2, text="Naive Annual Volatility: ")
NAV_label.grid(row=2, column=1, padx=10, pady=10)
NAV_var = StringVar()

NAV_textbox = Entry(frame2, width=35, borderwidth=5,
                    textvariable=NAV_var, state=DISABLED)
NAV_textbox.grid(row=2, column=2, columnspan=2, padx=10, pady=10)

NSR_label = Label(frame2, text="Naive Sharpe Ratio: ")
NSR_label.grid(row=3, column=1, padx=10, pady=10)
NSR_var = StringVar()

NSR_textbox = Entry(frame2, width=35, borderwidth=5,
                    textvariable=NSR_var, state=DISABLED)
NSR_textbox.grid(row=3, column=2, columnspan=2, padx=10, pady=10)

EAR_label = Label(frame2, text="Expected Annual Returns: ")
EAR_label.grid(row=4, column=1, padx=10, pady=10)
EAR_var = StringVar()

EAR_textbox = Entry(frame2, width=35, borderwidth=5,
                    textvariable=EAR_var, state=DISABLED)
EAR_textbox.grid(row=4, column=2, columnspan=2, padx=10, pady=10)

AV_label = Label(frame2, text="Annual Volatility: ")
AV_label.grid(row=5, column=1, padx=10, pady=10)
AV_var = StringVar()

AV_textbox = Entry(frame2, width=35, borderwidth=5,
                   textvariable=AV_var, state=DISABLED)
AV_textbox.grid(row=5, column=2, columnspan=2, padx=10, pady=10)

SR_label = Label(frame2, text="Sharpe Ratio: ")
SR_label.grid(row=6, column=1, padx=10, pady=10)

SR = StringVar()
SR_textbox = Entry(frame2, width=35, borderwidth=5,
                   textvariable=SR, state=DISABLED)
SR_textbox.grid(row=6, column=2, columnspan=2, padx=10, pady=10)


# new code
root.mainloop()
