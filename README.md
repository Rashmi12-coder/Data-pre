# Data-pre
# Task 1: Data Cleaning & Preprocessing
# Author: Rashmi Risha.J

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Step 1: Load the Dataset
df = pd.read_csv('data/your_dataset.csv')  # Make sure this path is correct
print("Initial Data Info:\n", df.info())
print("\nData Preview:\n", df.head())

# Step 2: Handle Missing Values
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Step 3: Convert Categorical to Numerical
# One-Hot Encoding
non_ordinal = ['CategoryColumn']  # Replace with actual column names
df = pd.get_dummies(df, columns=non_ordinal, drop_first=True)

# Label Encoding
ordinal = ['OrdinalColumn']  # Replace with actual column names
le = LabelEncoder()
for col in ordinal:
    df[col] = le.fit_transform(df[col])

# Step 4: Standardize Numerical Features
scaler = StandardScaler()
to_scale = ['NumColumn1', 'NumColumn2']  # Replace with actual numerical column names
df[to_scale] = scaler.fit_transform(df[to_scale])

# Step 5: Detect & Remove Outliers (IQR)
sns.boxplot(data=df[to_scale])
plt.title("Boxplot for Outliers")
plt.savefig('visuals/boxplot_outliers.png')
plt.show()

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Final Check
print("\nCleaned Data Info:\n", df.info())
print(df.head())
