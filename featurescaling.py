import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Step 1: Create the DataFrame
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [35, 45, 50, 65, 85]
}
df = pd.DataFrame(data)

# Step 2: Apply StandardScaler
scaler = StandardScaler()
standard_scaled_data = scaler.fit_transform(df)
standard_scaled_df = pd.DataFrame(standard_scaled_data, columns=['Hours_standard', 'Marks_standard'])

# Step 3: Apply MinMaxScaler
minmax_scaler = MinMaxScaler()
minmax_scaled_data = minmax_scaler.fit_transform(df)
minmax_scaled_df = pd.DataFrame(minmax_scaled_data, columns=['Hours_minmax', 'Marks_minmax'])

# Display both scaled DataFrames
print("Standard Scaled Data:")
print(standard_scaled_df)

print("\nMinMax Scaled Data:")
print(minmax_scaled_df)
