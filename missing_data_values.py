import pandas as pd
import numpy as np


data = {
    'Name': ['Alice', 'Bob', 'Charlie', np.nan, 'Eve'],
    'Age': [25, 30, np.nan, 35, 40],
    'Salary': [50000, 60000, 75000, np.nan, 85000]
}

df = pd.DataFrame(data)


print("Original DataFrame:")
print(df)


dropped_df = df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(dropped_df)


filled_df = df.fillna({'Name': 'Unknown', 'Age': df['Age'].mean(), 'Salary': df['Salary'].median()})
print("\nDataFrame after filling missing values:")
print(filled_df)


ffill_df = df.fillna(method='ffill')
print("\nDataFrame after forward fill:")
print(ffill_df)


bfill_df = df.fillna(method='bfill')
print("\nDataFrame after backward fill:")
print(bfill_df)

interpolated_df = df.interpolate()
print("\nDataFrame after interpolating missing values:")
print(interpolated_df)


missing_values = df.isnull().sum()
print("\nCount of missing values in each column:")
print(missing_values)


zero_filled_df = df.fillna(0)
print("\nDataFrame after filling missing values with 0:")
print(zero_filled_df)


df['Salary'] = df['Salary'].fillna(df['Age'].apply(lambda x: 50000 if x > 30 else 40000))
print("\nDataFrame after conditional replacement:")
print(df)


column_filled_df = df.copy()
column_filled_df['Age'] = column_filled_df['Age'].fillna(column_filled_df['Age'].mean())
column_filled_df['Name'] = column_filled_df['Name'].fillna("Unknown")
print("\nDataFrame after column-wise replacement:")
print(column_filled_df)


dropped_columns_df = df.dropna(axis=1)
print("\nDataFrame after dropping columns with missing values:")
print(dropped_columns_df)

from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy='mean')
df['Age'] = imputer.fit_transform(df[['Age']])
df['Salary'] = imputer.fit_transform(df[['Salary']])
print("\nDataFrame after using SimpleImputer:")
print(df)


custom_df = df.copy()
custom_df['Name'] = custom_df['Name'].fillna("Guest")
custom_df['Salary'] = custom_df['Salary'].fillna(45000)
print("\nDataFrame after applying custom logic:")
print(custom_df)
