import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("data.csv")

# Drop rows with invalid price
df = df[df["price"] > 0].copy()

# Select features
features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", 
            "waterfront", "view", "condition", "sqft_above", 
            "sqft_basement", "yr_built", "yr_renovated"]

X = df[features]
y = np.log1p(df["price"])  # log-transform target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
r2 = r2_score(y_test, y_pred)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))

# Convert back to price scale
y_pred_actual = np.expm1(y_pred)
y_test_actual = np.expm1(y_test)
rmse_price = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))

print("R² (log scale):", round(r2 * 100, 2), "%")
print("RMSE (log scale):", round(rmse_log, 4))
print("RMSE (price scale):", round(rmse_price, 2))
print("\nSample Predictions:", y_pred_actual[:5])
print("Sample Actuals:", y_test_actual[:5])

# Cross-validation (3-fold to avoid instability)
scores = cross_val_score(model, X, y, cv=3, scoring="r2", n_jobs=-1)
print("\nAverage CV R²:", round(scores.mean() * 100, 2), "%")
print("CV R² scores:", [round(score * 100, 2) for score in scores])