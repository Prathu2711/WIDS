
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("linear_regression_dataset.csv")

X = df.drop(columns=["y"]).values
y = df["y"].values

# Add intercept
X = np.c_[np.ones(X.shape[0]), X]

# -------- Question 7 --------
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

model = LinearRegression(fit_intercept=True)
model.fit(df.drop(columns=["y"]), y)

print("Closed-form coefficients:", beta_hat)
print("Sklearn intercept:", model.intercept_)
print("Sklearn coefficients:", model.coef_)

# -------- Question 8 --------
y_hat = X @ beta_hat
residuals = y - y_hat

plt.figure()
plt.scatter(y_hat, residuals)
plt.axhline(0, color='red')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()

# -------- Question 9 --------
plt.figure()
stats.probplot(residuals, plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.show()

# -------- Question 11 --------
H = X @ np.linalg.inv(X.T @ X) @ X.T
leverage = np.diag(H)
print("Maximum leverage:", leverage.max())

# -------- Question 14 --------
n = 500
x1 = np.random.randn(n)
z = np.random.randn(n)
x2 = x1 + 0.9 * z

X_mc = np.c_[np.ones(n), x1, x2]
condition_number = np.linalg.cond(X_mc.T @ X_mc)
print("Condition number:", condition_number)
