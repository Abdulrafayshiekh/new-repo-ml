import pandas as pd
from sklearn.model_selection import train_test_split

# Step 1: Create the DataFrame
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [35, 45, 50, 65, 85]
}
df = pd.DataFrame(data)

# Step 2: Define features and target
X = df[['Hours']]       # Independent variable
y = df['Marks']         # Dependent variable

# Step 3: Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Display the results
print("X_train:\n", X_train)
print("\nX_test:\n", X_test)
print("\ny_train:\n", y_train)
print("\ny_test:\n", y_test)
