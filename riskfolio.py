import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt

# Define the assets for the portfolio
assets = ["AAPL", "MSFT", "TSLA", "AMZN"]
start_date = "2023-01-01"
end_date = "2024-01-01"

# Download historical price data
data = yf.download(assets, start=start_date, end=end_date)

# Use "Close" prices instead of "Adj Close" to avoid KeyError
returns = data["Close"].pct_change().dropna()

# Initialize Riskfolio-Lib Portfolio object
port = rp.Portfolio(returns=returns)

# Compute mean returns and covariance matrix using historical method
port.assets_stats(method_mu="hist", method_cov="hist")

# Perform Mean-Variance optimization (Markowitz model)
w = port.optimization(model="Classic", rm="MV", obj="Sharpe", rf=0, l=0)

# Plot the optimized portfolio allocation
fig, ax = plt.subplots(figsize=(10, 5))
rp.plot_pie(w=w, title="Optimized Portfolio Allocation (Markowitz Model)", ax=ax)
plt.show()

# Print portfolio weights
print("Optimized Portfolio Weights:")
print(w)
