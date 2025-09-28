import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
# Load and prepare data
data = pd.read_csv("score.csv")
X = data[['Hours']]
y = data['Scores']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for user input
hours = float(input("Enter hours studied: "))
input_df = pd.DataFrame({'Hours': [hours]})
predicted = model.predict(input_df)
capped_score = min(round(predicted[0], 2), 100)
print("The predicted score is:", capped_score)

# Predict for all rows and add new column
data['Predicted_Score'] = np.clip(model.predict(X).round(2), None, 100)

# Export to CSV
data.to_csv("score_with_predictions.csv", index=False)
