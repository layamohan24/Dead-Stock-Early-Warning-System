import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/raw/dead_stock_dataset.csv")

# Convert Seasonality into numbers
seasonality_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["Seasonality"] = df["Seasonality"].map(seasonality_mapping)


# Features used by the ML models
features = [
    "Days_Since_Last_Sale",
    "Sales_Velocity",
    "Inventory_Quantity",
    "Product_Age",
    "Price",
    "Discount",
    "Seasonality",
    "Customer_Interest"
]

# Input features
X = df[features]

# Target
y = df["Dead_Stock"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("=" * 50)
print("PREPROCESSING COMPLETED")
print("=" * 50)

print("\nTotal samples:", len(df))

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nNumber of features:", X.shape[1])

print("\nFeatures:")
for feature in features:
    print("-", feature)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# Save processed data
train_data = X_train.copy()
train_data["Dead_Stock"] = y_train

test_data = X_test.copy()
test_data["Dead_Stock"] = y_test

train_data.to_csv(
    "data/processed/train_data.csv",
    index=False
)

test_data.to_csv(
    "data/processed/test_data.csv",
    index=False
)

print("\nProcessed datasets saved successfully!")