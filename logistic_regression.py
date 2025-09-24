from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Training data
X = [[1], [2], [3], [4], [5], [6]]
y = [0, 0, 0, 1, 1, 1]  # 0 = Fail, 1 = Pass

# Train model
model = LogisticRegression()
model.fit(X, y)

# Input for prediction
hours = float(input("Enter hours studied: "))
result = model.predict([[hours]])

# Print result
print(f"Hours Studied: {hours}")
print(f"Predicted Result: {'Pass' if result[0] == 1 else 'Fail'}")

# Chart 2: Color-coded prediction
plt.figure()
color = 'green' if result[0] == 1 else 'red'
label = 'Pass' if result[0] == 1 else 'Fail'
plt.scatter(hours, result, color=color, s=300, edgecolors='black', label=label)
plt.xlabel('Hours Studied')
plt.ylabel('Prediction')
plt.title('Prediction Outcome')
plt.legend()
plt.grid(True)
plt.show()
