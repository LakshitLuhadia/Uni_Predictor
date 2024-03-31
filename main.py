import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

df = pd.read_csv("admission_predict.csv")
df.rename(
    columns={
        "GRE Score": "GRE",
        "TOEFL Score": "TOEFL",
        "LOR ": "LOR",
        "Chance of Admit ": "Probability",
    },
    inplace=True,
)
df.drop("Serial No.", axis=1, inplace=True)

df["Composite_Score"] = df["GRE"] + df["TOEFL"]

df.fillna(df.mean(), inplace=True)

X = df.drop("Probability", axis=1)
y = df["Probability"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])

param_grid = {"ridge__alpha": np.logspace(-4, 4, 20)}

grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Train score: {grid.score(X_train, y_train):.2f}")
print(f"Test score: {grid.score(X_test, y_test):.2f}")

from joblib import dump

grid.fit(X_train, y_train)

dump(grid, "admission_predictor_model.joblib")
