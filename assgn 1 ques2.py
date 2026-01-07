
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import shap

# Load dataset
df = pd.read_csv("salary_dataset.csv")

# ---------------- EDA ----------------
print(df.describe())

# ---------------- Missing values ----------------
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

# ---------------- Features ----------------
X = df.drop(columns=["salary"])
y = df["salary"]

categorical = X.select_dtypes(include=["object"]).columns
numerical = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical),
        ("num", "passthrough", numerical),
    ]
)

# ---------------- Train-test split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df["gender"]
)

# ---------------- Model ----------------
model = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("regressor", LinearRegression())]
)

model.fit(X_train, y_train)

# ---------------- Evaluation ----------------
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# ---------------- Fairness metrics ----------------
test_df = X_test.copy()
test_df["y_true"] = y_test.values
test_df["y_pred"] = y_pred

for g in test_df["gender"].unique():
    grp = test_df[test_df["gender"] == g]
    print(f"{g} MAE:", mean_absolute_error(grp["y_true"], grp["y_pred"]))

# ---------------- Residual t-test ----------------
male_res = test_df[test_df["gender"] == "Male"]["y_true"] - test_df[test_df["gender"] == "Male"]["y_pred"]
female_res = test_df[test_df["gender"] == "Female"]["y_true"] - test_df[test_df["gender"] == "Female"]["y_pred"]
print("T-test:", stats.ttest_ind(male_res, female_res))

# ---------------- SHAP ----------------
explainer = shap.Explainer(model["regressor"], model["preprocessor"].transform(X_train))
shap_values = explainer(model["preprocessor"].transform(X_test))
shap.summary_plot(shap_values, show=False)
plt.show()
