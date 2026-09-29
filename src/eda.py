import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/raw/dead_stock_dataset.csv")

# Create output folder
import os
os.makedirs("data/processed/graphs", exist_ok=True)


# --------------------------------------------------
# 1. Dead Stock Distribution
# --------------------------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="Dead_Stock"
)

plt.title("Dead Stock Distribution")
plt.xlabel("Dead Stock (0 = No, 1 = Yes)")
plt.ylabel("Number of Products")

plt.savefig(
    "data/processed/graphs/dead_stock_distribution.png"
)

plt.show()


# --------------------------------------------------
# 2. Days Since Last Sale vs Dead Stock
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Dead_Stock",
    y="Days_Since_Last_Sale"
)

plt.title("Days Since Last Sale vs Dead Stock")
plt.xlabel("Dead Stock")
plt.ylabel("Days Since Last Sale")

plt.savefig(
    "data/processed/graphs/days_since_sale.png"
)

plt.show()


# --------------------------------------------------
# 3. Sales Velocity vs Dead Stock
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Dead_Stock",
    y="Sales_Velocity"
)

plt.title("Sales Velocity vs Dead Stock")
plt.xlabel("Dead Stock")
plt.ylabel("Sales Velocity")

plt.savefig(
    "data/processed/graphs/sales_velocity.png"
)

plt.show()


# --------------------------------------------------
# 4. Inventory vs Dead Stock
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Dead_Stock",
    y="Inventory_Quantity"
)

plt.title("Inventory Quantity vs Dead Stock")
plt.xlabel("Dead Stock")
plt.ylabel("Inventory Quantity")

plt.savefig(
    "data/processed/graphs/inventory.png"
)

plt.show()


# --------------------------------------------------
# 5. Customer Interest vs Dead Stock
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Dead_Stock",
    y="Customer_Interest"
)

plt.title("Customer Interest vs Dead Stock")
plt.xlabel("Dead Stock")
plt.ylabel("Customer Interest")

plt.savefig(
    "data/processed/graphs/customer_interest.png"
)

plt.show()


# --------------------------------------------------
# 6. Product Age vs Dead Stock
# --------------------------------------------------

plt.figure(figsize=(7, 5))

sns.boxplot(
    data=df,
    x="Dead_Stock",
    y="Product_Age"
)

plt.title("Product Age vs Dead Stock")
plt.xlabel("Dead Stock")
plt.ylabel("Product Age (days)")

plt.savefig(
    "data/processed/graphs/product_age.png"
)

plt.show()


# --------------------------------------------------
# 7. Correlation Heatmap
# --------------------------------------------------

plt.figure(figsize=(10, 7))

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
)

correlation = numeric_columns.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.savefig(
    "data/processed/graphs/correlation_heatmap.png"
)

plt.show()


print("EDA completed successfully!")
print("Graphs saved inside:")
print("data/processed/graphs/")