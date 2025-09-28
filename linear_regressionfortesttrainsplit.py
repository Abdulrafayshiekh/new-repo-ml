import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv("score.csv")

# Separate features and target
X = data[['Hours']]
y = data['Scores']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict score for user input
try:
    hours = float(input("Enter hours studied: "))
    input_scaled = scaler.transform([[hours]])
    predicted_score = model.predict(input_scaled)[0]
    capped_score = np.clip(round(predicted_score, 2), 0, 100)
    print("📊 Predicted score:", capped_score)
except ValueError:
    print("⚠️ Invalid input. Please enter a numeric value.")

# Predict on test set
y_pred_test = model.predict(X_test_scaled)
y_pred_test_capped = np.clip(np.round(y_pred_test, 2), 0, 100)

# Create test results DataFrame
test_results = X_test.copy()
test_results['Actual_Score'] = y_test
test_results['Predicted_Score'] = y_pred_test_capped

# Export predictions to CSV
test_results.to_csv("test_predictions.csv", index=False)
print("✅ Test predictions saved to 'test_predictions.csv'")
