import pandas as pd


# Load clustered dataset
df = pd.read_csv(
    "data/processed/clustered_products.csv"
)


# Calculate average characteristics
summary = df.groupby("Cluster").agg({
    "Days_Since_Last_Sale": "mean",
    "Sales_Velocity": "mean",
    "Inventory_Quantity": "mean",
    "Product_Age": "mean",
    "Customer_Interest": "mean",
    "Seasonality": "mean"
})


print("=" * 70)
print("CLUSTER CHARACTERISTICS")
print("=" * 70)

print(summary.round(2))


# --------------------------------------------------
# Identify clusters
# --------------------------------------------------

# Highest sales velocity → Fast Sellers
fast_cluster = summary["Sales_Velocity"].idxmax()


# Lowest sales velocity → Slow Sellers
slow_cluster = summary["Sales_Velocity"].idxmin()


# Highest days since last sale + high inventory
risk_score = (
    summary["Days_Since_Last_Sale"]
    + summary["Inventory_Quantity"] / 5
    - summary["Sales_Velocity"] * 10
    - summary["Customer_Interest"] / 10
)

risk_cluster = risk_score.idxmax()


# Remaining cluster → Seasonal
remaining_clusters = set(summary.index) - {
    fast_cluster,
    slow_cluster,
    risk_cluster
}

seasonal_cluster = list(remaining_clusters)[0]


# --------------------------------------------------
# Create labels
# --------------------------------------------------

cluster_labels = {
    fast_cluster: "Fast Sellers",
    slow_cluster: "Slow Sellers",
    seasonal_cluster: "Seasonal Products",
    risk_cluster: "Dead-Stock Risk"
}


df["Cluster_Name"] = df["Cluster"].map(cluster_labels)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLUSTER LABELS")
print("=" * 70)

for cluster, label in cluster_labels.items():
    print(f"Cluster {cluster} → {label}")


print("\nProduct distribution:")

print(
    df["Cluster_Name"].value_counts()
)


# --------------------------------------------------
# Save final clustered dataset
# --------------------------------------------------

df.to_csv(
    "data/processed/clustered_products_labeled.csv",
    index=False
)


print("\nSaved:")
print(
    "data/processed/clustered_products_labeled.csv"
)