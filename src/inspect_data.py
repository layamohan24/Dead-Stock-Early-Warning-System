import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/dead_stock_dataset.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

# Number of rows and columns
print("\nShape:")
print(df.shape)

# Column names
print("\nColumns:")
print(df.columns.tolist())

# First 5 products
print("\nFirst 5 rows:")
print(df.head())

# Data types
print("\nData types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Dead stock distribution
print("\nDead Stock distribution:")
print(df["Dead_Stock"].value_counts())

# Dead stock percentage
print("\nDead Stock percentage:")
print(df["Dead_Stock"].value_counts(normalize=True) * 100)