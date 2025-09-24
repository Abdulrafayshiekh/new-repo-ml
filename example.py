import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


# Create a larger DataFrame with Gender
data = {
    'StudentID': list(range(101, 121)),
    'Gender': ['Male', 'Female'] * 10,  # Alternating genders
    'Hours_Studied': [2, 4, 1, 5, 3, 6, 2.5, 4.5, 3.5, 5.5, 3.2, 4.8, 2.1, 1.5, 5.2, 3.8, 2.7, 4.3, 1.8, 6.1],
    'Attendance': [80, 90, 60, 95, 85, 100, 70, 88, 92, 98, 86, 93, 75, 65, 97, 89, 78, 91, 68, 99],
    'Assignment_Score': [65, 70, 50, 85, 60, 90, 55, 75, 80, 88, 68, 82, 58, 52, 87, 73, 61, 79, 54, 92],
    'Quiz_Score': [70, 75, 55, 90, 65, 95, 60, 80, 85, 92, 72, 88, 62, 57, 91, 77, 66, 84, 59, 96],
    'Project_Score': [60, 65, 50, 80, 55, 85, 52, 70, 75, 82, 60, 78, 54, 49, 81, 67, 56, 74, 51, 86],
    'Final_Marks': [50, 60, 40, 75, 55, 85, 45, 65, 70, 80, 58, 73, 48, 42, 78, 62, 52, 68, 44, 88]
}

df = pd.DataFrame(data)
df.to_csv('student_data.csv', index=False)

# Create a copy for safe transformation
df_original = df.copy()

# Label encode Gender (categorical)
le = LabelEncoder()
df_original['gender_encoded'] = le.fit_transform(df_original['Gender'])

# Do NOT label encode numeric columns — use scalers instead
# You can now apply StandardScaler or MinMaxScaler to Attendance, Final_Marks, etc.
print(df_original[['StudentID', 'Gender','gender_encoded']])

sc=StandardScaler()
scaled_data=sc.fit_transform(df_original[['Attendance','Final_Marks']])
standard_scaled_df = pd.DataFrame(scaled_data, columns=['Hours_standard', 'Marks_standard'])
print("\nStandard Scaled Data:")
print(standard_scaled_df)
mm=MinMaxScaler()
minmax_scaled_data=mm.fit_transform(df_original[['Attendance','Final_Marks']])
minmax_scaled_df = pd.DataFrame(minmax_scaled_data, columns=['Hours_minmax', 'Marks_minmax'])
print("\nMinMax Scaled Data:")
print(minmax_scaled_df)
# Note: Do not label encode numeric columns like Attendance, Final_Marks, etc.

X= df_original[['Hours_Studied']]
y = df_original['Final_Marks']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nX_train:\n", X_train)
print("\nX_test:\n", X_test)

print("\ny_train:\n", y_train)
print("\ny_test:\n", y_test)
