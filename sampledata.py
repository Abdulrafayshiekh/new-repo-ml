import pandas as pd
import numpy as np

# Create sample data
data = {
    'Name': ['Ali', 'Sara', 'John', 'Meera', 'Kamal', 'Zain', 'Ayesha', 'Bilal', 'Nida', 'Omar', 'Hina', 'Usman', 'Fatima', 'Kashif', 'Rabia'],
    'City': ['Lahore', 'Karachi', 'Rawalpindi', 'Islamabad', 'Faisalabad', 'Multan', 'Lahore', 'Karachi', 'Peshawar', 'Quetta', 'Hyderabad', 'Multan', 'Islamabad', 'Faisalabad', 'Lahore'],
    'Passed': ['Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes'],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female']
}


# Convert to DataFrame
df = pd.DataFrame(data)

# Show the DataFrame
print("Sample Data:")
print(df)
# print("Missing values summary by column")

# print(df.isnull().sum())
# option 2 drop na
# df_cleaned = df.dropna()
# # print it 
# print(df_cleaned)
# select a column and fill na with a value
# df['AGE'] = df['AGE'].fillna(df['AGE'].mean())  

# Save to CSV
df.to_csv('sample_data.csv', index=False)
import pandas as pd
import numpy as np

# Create a larger dataset with some missing values
data = {
    'Name': ['Ali', 'Sara', 'John', 'Meera', 'Zain', 'Ayesha', 'Bilal', 'Nida', 'Omar', 'Hina', 'Usman', 'Fatima', 'Kashif', None, 'Rabia'],
    'Age': [25, 30, 22, 28, 35, None, 40, 27, 33, 29, None, 31, 26, 24, 38],
    'Salary': [50000, 60000, 45000, 52000, 58000, 61000, None, 49000, 53000, None, 47000, 62000, 51000, 55000, None]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print("Expanded Employee Data with Missing Values:")
print(df)

# Save to CSV
df.to_csv('expanded_employee_data.csv', index=False)
