from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the dataset
df = pd.read_csv('sample_data.csv')

# Keep a copy of the original data for comparison
df_original = df.copy()

# Initialize LabelEncoder
le = LabelEncoder()

# Apply label encoding to Gender and Passed columns
df_original['gender_encoded'] = le.fit_transform(df_original['Gender'])
df_original['passed_encoded'] = le.fit_transform(df_original['Passed'])

# Print the encoded data
print("\nLabel encoded data")
print(df_original[['Name', 'Gender', 'gender_encoded', 'Passed', 'passed_encoded']])
