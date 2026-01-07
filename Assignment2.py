 Assignment 2: Kalman Filtered Trading Strategy for MSFT
# ------------------------------------------------------
# This script fetches data, engineers features, applies a Kalman Filter,
# integrates a machine learning model, generates trading signals,
# and evaluates performance via backtesting.

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from sklearn.linear_model import Ridge

# ---------------- Data Fetching ----------------
data = yf.download("MSFT", start="2015-01-01", end="2024-01-01")
data = data.dropna()

# ---------------- Feature Engineering ----------------
data["log_return"] = np.log(data["Adj Close"]).diff()
data["ma_5"] = data["Adj Close"].rolling(5).mean()
data["ma_20"] = data["Adj Close"].rolling(20).mean()
data["ma_60"] = data["Adj Close"].rolling(60).mean()
data["roc"] = data["Adj Close"].pct_change(5)
data["volatility"] = data["log_return"].rolling(20).std()

data = data.dropna()

features = ["ma_5", "ma_20", "ma_60", "roc", "volatility"]
X = data[features].values
y = data["log_return"].values

# ---------------- Kalman Filter ----------------
kf = KalmanFilter(
    transition_matrices=np.eye(len(features)),
    observation_matrices=X[:, np.newaxis, :],
    transition_covariance=0.01 * np.eye(len(features)),
    observation_covariance=1.0
)

state_means, _ = kf.filter(y)

# ---------------- ML Model ----------------
model = Ridge(alpha=1.0)
model.fit(state_means[:-1], y[1:])

predicted_returns = model.predict(state_means)

# ---------------- Trading Signals ----------------
threshold = 0.0005
signals = np.where(predicted_returns > threshold, 1,
           np.where(predicted_returns < -threshold, -1, 0))

# ---------------- Backtesting ----------------
data["signal"] = signals
data["strategy_return"] = data["signal"].shift(1) * data["log_return"]
data["equity_curve"] = (1 + data["strategy_return"].fillna(0)).cumprod()
data["buy_hold"] = (1 + data["log_return"].fillna(0)).cumprod()

# ---------------- Evaluation ----------------
sharpe = np.sqrt(252) * data["strategy_return"].mean() / data["strategy_return"].std()
drawdown = (data["equity_curve"] / data["equity_curve"].cummax() - 1).min()

print("Sharpe Ratio:", sharpe)
print("Max Drawdown:", drawdown)

# ---------------- Plots ----------------
plt.figure()
plt.plot(data["equity_curve"], label="Strategy")
plt.plot(data["buy_hold"], label="Buy & Hold")
plt.legend()
plt.title("Equity Curve Comparison")
plt.show()
