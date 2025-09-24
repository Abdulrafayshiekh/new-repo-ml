from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Sample data
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'Marks': [35, 40, 50, 55, 60, 65, 70, 75, 85]
}
df = pd.DataFrame(data)

# Train model
X = df[['Hours']]
y = df['Marks']
model = LinearRegression()
model.fit(X, y)

# Take input
hours = float(input("Enter hours studied: "))
predicted = model.predict([[hours]])

# Print result
print(f"Hours Studied: {hours}")
print(f"Predicted Marks: {round(predicted[0], 2)}")

# Plot
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.scatter(hours, predicted, color='green', marker='*', s=200, label='Predicted Point')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Obtained')
plt.title('Study Hours vs Marks')
plt.legend()
plt.grid(True)
plt.show()
