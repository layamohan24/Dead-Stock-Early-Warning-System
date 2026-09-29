import pandas as pd
import numpy as np

# For reproducible results
np.random.seed(42)

# Number of products
N = 5000

# Product categories
categories = [
    "T-Shirt",
    "Jeans",
    "Hoodie",
    "Jacket",
    "Shoes",
    "Dress",
    "Shirt",
    "Accessories"
]

# Create product IDs
product_ids = [f"P{i:04d}" for i in range(1, N + 1)]

# Generate product categories
product_categories = np.random.choice(categories, N)

# Product names based on category
product_names = [
    f"{category} {i:04d}"
    for i, category in enumerate(product_categories, start=1)
]

# Days since last sale
days_since_last_sale = np.random.randint(0, 121, N)

# Sales velocity (units sold per day)
sales_velocity = np.round(
    np.random.gamma(shape=2, scale=2, size=N),
    2
)

# Inventory quantity
inventory_quantity = np.random.randint(5, 301, N)

# Product age in days
product_age = np.random.randint(7, 366, N)

# Product price
price = np.random.choice(
    [799, 999, 1299, 1499, 1999, 2499, 2999, 3999, 4999],
    N
)

# Discount percentage
discount = np.random.choice(
    [0, 5, 10, 15, 20, 30, 40, 50],
    N
)

# Seasonality
seasonality = np.random.choice(
    ["Low", "Medium", "High"],
    N,
    p=[0.3, 0.45, 0.25]
)

# Customer interest (0-100)
customer_interest = np.random.randint(0, 101, N)


# ---------------------------------------------------
# Create a realistic dead-stock risk score
# ---------------------------------------------------

risk_score = (
    (days_since_last_sale / 120) * 0.30
    + ((10 - np.minimum(sales_velocity, 10)) / 10) * 0.25
    + (inventory_quantity / 300) * 0.20
    + (product_age / 365) * 0.10
    + ((100 - customer_interest) / 100) * 0.15
)

# Add some randomness
risk_score += np.random.normal(0, 0.08, N)

# Dead stock
dead_stock = (risk_score > 0.50).astype(int)


# Create DataFrame
df = pd.DataFrame({
    "Product_ID": product_ids,
    "Product_Name": product_names,
    "Category": product_categories,
    "Days_Since_Last_Sale": days_since_last_sale,
    "Sales_Velocity": sales_velocity,
    "Inventory_Quantity": inventory_quantity,
    "Product_Age": product_age,
    "Price": price,
    "Discount": discount,
    "Seasonality": seasonality,
    "Customer_Interest": customer_interest,
    "Dead_Stock": dead_stock
})


# Save dataset
df.to_csv(
    "data/raw/dead_stock_dataset.csv",
    index=False
)


# Display information
print("Dataset created successfully!")
print(f"Number of products: {len(df)}")
print("\nFirst 10 rows:")
print(df.head(10))

print("\nDataset shape:")
print(df.shape)

print("\nDead stock distribution:")
print(df["Dead_Stock"].value_counts())

print("\nDataset saved to:")
print("data/raw/dead_stock_dataset.csv")