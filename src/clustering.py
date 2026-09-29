import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Load dataset
df = pd.read_csv("data/raw/dead_stock_dataset.csv")


# Convert seasonality to numbers
seasonality_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

df["Seasonality"] = df["Seasonality"].map(seasonality_mapping)


# Features used for clustering
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

X = df[features]


# --------------------------------------------------
# Standardize the data
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Elbow Method
# --------------------------------------------------

inertia = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")

plt.tight_layout()

plt.savefig(
    "data/processed/graphs/elbow_method.png"
)

plt.show()


# --------------------------------------------------
# Create 4 clusters
# --------------------------------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# --------------------------------------------------
# Analyze clusters
# --------------------------------------------------

cluster_summary = df.groupby("Cluster")[features].mean()

print("=" * 70)
print("CLUSTER SUMMARY")
print("=" * 70)

print(cluster_summary)


print("\nNumber of products in each cluster:")

print(
    df["Cluster"].value_counts().sort_index()
)


# --------------------------------------------------
# Save clustered dataset
# --------------------------------------------------

df.to_csv(
    "data/processed/clustered_products.csv",
    index=False
)


print("\nClustered dataset saved successfully!")

print(
    "data/processed/clustered_products.csv"
)