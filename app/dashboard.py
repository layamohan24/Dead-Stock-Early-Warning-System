import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# LOAD MODEL AND DATA
# --------------------------------------------------

model = joblib.load("models/random_forest.pkl")
df = pd.read_csv("data/raw/dead_stock_dataset.csv")

cluster_df = pd.read_csv(
    "data/processed/clustered_products_labeled.csv"
)
# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Dead Stock Early Warning System",
    page_icon="📦",
    layout="wide"
)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📦 Dead Stock Early Warning System")
st.write(
    "Predict inventory risk before products become dead stock."
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("🔍 Product Details")

days_since_last_sale = st.sidebar.number_input(
    "Days Since Last Sale",
    min_value=0,
    max_value=120,
    value=30
)

sales_velocity = st.sidebar.number_input(
    "Sales Velocity",
    min_value=0.0,
    max_value=20.0,
    value=5.0
)

inventory_quantity = st.sidebar.number_input(
    "Inventory Quantity",
    min_value=0,
    max_value=500,
    value=100
)

product_age = st.sidebar.number_input(
    "Product Age (days)",
    min_value=1,
    max_value=500,
    value=100
)

price = st.sidebar.number_input(
    "Price",
    min_value=0,
    max_value=10000,
    value=1499
)

discount = st.sidebar.number_input(
    "Discount (%)",
    min_value=0,
    max_value=100,
    value=10
)

seasonality = st.sidebar.selectbox(
    "Seasonality",
    ["Low", "Medium", "High"]
)

seasonality_map = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

customer_interest = st.sidebar.slider(
    "Customer Interest",
    0,
    100,
    50
)

predict_button = st.sidebar.button(
    "🔍 Predict Risk"
)

# --------------------------------------------------
# INVENTORY OVERVIEW
# --------------------------------------------------

st.subheader("📦 Inventory Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Products",
    len(df)
)

col2.metric(
    "Dead Stock Products",
    int(df["Dead_Stock"].sum())
)

col3.metric(
    "Average Inventory",
    f"{df['Inventory_Quantity'].mean():.0f}"
)

col4.metric(
    "Average Sales Velocity",
    f"{df['Sales_Velocity'].mean():.2f}"
)

st.divider()

# --------------------------------------------------
# INVENTORY ANALYTICS
# --------------------------------------------------

st.subheader("📊 Inventory Analytics")

col1, col2 = st.columns(2)

# Dead stock distribution
with col1:

    st.write("### Dead Stock Distribution")

    dead_stock_counts = df["Dead_Stock"].value_counts()

    chart_data = pd.DataFrame({
        "Status": [
            "Healthy",
            "Dead Stock"
        ],
        "Products": [
            dead_stock_counts.get(0, 0),
            dead_stock_counts.get(1, 0)
        ]
    })

    st.bar_chart(
        chart_data.set_index("Status")
    )

# Category distribution
with col2:

    st.write("### Products by Category")

    category_counts = df["Category"].value_counts()

    st.bar_chart(category_counts)

# --------------------------------------------------
# CATEGORY SUMMARY
# --------------------------------------------------

st.divider()

st.subheader("📋 Inventory Summary by Category")

category_summary = df.groupby("Category").agg({
    "Inventory_Quantity": "mean",
    "Sales_Velocity": "mean",
    "Customer_Interest": "mean",
    "Dead_Stock": "sum"
}).round(2)

category_summary = category_summary.rename(columns={
    "Inventory_Quantity": "Avg Inventory",
    "Sales_Velocity": "Avg Sales Velocity",
    "Customer_Interest": "Avg Customer Interest",
    "Dead_Stock": "Dead Stock Products"
})

st.dataframe(
    category_summary,
    use_container_width=True
)

st.divider()

# --------------------------------------------------
# PRODUCT PREDICTION
# --------------------------------------------------

if predict_button:

    product = pd.DataFrame([{
        "Days_Since_Last_Sale": days_since_last_sale,
        "Sales_Velocity": sales_velocity,
        "Inventory_Quantity": inventory_quantity,
        "Product_Age": product_age,
        "Price": price,
        "Discount": discount,
        "Seasonality": seasonality_map[seasonality],
        "Customer_Interest": customer_interest
    }])

    # Model prediction
    prediction = model.predict(product)[0]

    probability = model.predict_proba(product)[0][1]

    # Risk classification
    if probability < 0.30:
        risk = "LOW"

    elif probability < 0.60:
        risk = "MEDIUM"

    else:
        risk = "HIGH"

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.subheader("🔮 Inventory Health Prediction")

    col1, col2 = st.columns(2)

    col1.metric(
        "Dead Stock Probability",
        f"{probability * 100:.2f}%"
    )

    col2.metric(
        "Risk Level",
        risk
    )

    # Risk message
    if risk == "HIGH":

        st.error(
            "⚠️ High risk: This product may become dead stock."
        )

    elif risk == "MEDIUM":

        st.warning(
            "⚠️ Medium risk: Monitor this product closely."
        )

    else:

        st.success(
            "✅ Low risk: Product appears healthy."
        )

    # --------------------------------------------------
    # RISK FACTORS
    # --------------------------------------------------

    st.subheader("⚠️ Risk Factors")

    factors = []

    if days_since_last_sale > 60:
        factors.append(
            "Product has not been sold for a long time."
        )

    if sales_velocity < 2:
        factors.append(
            "Sales velocity is very low."
        )

    if inventory_quantity > 200:
        factors.append(
            "Inventory quantity is very high."
        )

    if product_age > 250:
        factors.append(
            "Product has been in inventory for a long time."
        )

    if customer_interest < 30:
        factors.append(
            "Customer interest is low."
        )

    if discount < 10 and probability > 0.50:
        factors.append(
            "Current discount may be insufficient to improve sales."
        )

    if len(factors) == 0:

        st.write(
            "✅ No major risk factors detected."
        )

    else:

        for factor in factors:
            st.write("• " + factor)

    # --------------------------------------------------
    # RECOMMENDATION
    # --------------------------------------------------

    st.subheader("💡 Recommendation")

    if risk == "HIGH":

        st.info(
            "Consider increasing the discount, running a "
            "promotional campaign, or reducing future stock orders."
        )

    elif risk == "MEDIUM":

        st.info(
            "Monitor sales closely and consider promotional activity."
        )

    else:

        st.info(
            "Continue normal inventory monitoring."
        )

else:

    st.info(
        "Enter product details in the sidebar and click "
        "**Predict Risk**."
    )

# --------------------------------------------------
# PRODUCT SEARCH
# --------------------------------------------------

st.divider()

st.subheader("🔎 Search Product")

search_term = st.text_input(
    "Enter Product ID or Product Name"
)

if search_term:

    search_results = cluster_df[
        cluster_df["Product_ID"].str.contains(
            search_term,
            case=False,
            na=False
        )
        |
        cluster_df["Product_Name"].str.contains(
            search_term,
            case=False,
            na=False
        )
    ]

    if len(search_results) > 0:

        st.success(
            f"{len(search_results)} product(s) found."
        )

        search_columns = [
            "Product_ID",
            "Product_Name",
            "Category",
            "Days_Since_Last_Sale",
            "Sales_Velocity",
            "Inventory_Quantity",
            "Product_Age",
            "Price",
            "Discount",
            "Customer_Interest",
            "Cluster_Name"
        ]

        st.dataframe(
            search_results[search_columns],
            use_container_width=True
        )

    else:

        st.warning("No products found.")

# --------------------------------------------------
# K-MEANS CLUSTER ANALYSIS
# --------------------------------------------------

st.divider()

st.subheader("🔬 Product Segmentation")

cluster_counts = cluster_df["Cluster_Name"].value_counts()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Fast Sellers",
    cluster_counts.get("Fast Sellers", 0)
)

col2.metric(
    "Slow Sellers",
    cluster_counts.get("Slow Sellers", 0)
)

col3.metric(
    "Seasonal Products",
    cluster_counts.get("Seasonal Products", 0)
)

col4.metric(
    "Dead-Stock Risk",
    cluster_counts.get("Dead-Stock Risk", 0)
)

st.write("### Product Segmentation Distribution")

st.bar_chart(cluster_counts)

st.write("### Segmented Products")

display_columns = [
    "Product_ID",
    "Product_Name",
    "Category",
    "Sales_Velocity",
    "Inventory_Quantity",
    "Customer_Interest",
    "Cluster_Name"
]

st.dataframe(
    cluster_df[display_columns],
    use_container_width=True
)

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

st.divider()

st.subheader("🤖 Machine Learning Model Comparison")

comparison = pd.read_csv(
    "data/processed/model_comparison.csv"
)

st.dataframe(
    comparison,
    use_container_width=True
)

st.write("### Model Performance")

performance_data = comparison.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

st.bar_chart(performance_data)

st.info(
    "Three classification models were trained and compared: "
    "Logistic Regression, Decision Tree, and Random Forest."
)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.divider()

st.subheader("📌 Important Risk Factors")

feature_names = [
    "Days Since Last Sale",
    "Sales Velocity",
    "Inventory Quantity",
    "Product Age",
    "Price",
    "Discount",
    "Seasonality",
    "Customer Interest"
]

importance_data = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance_data = importance_data.sort_values(
    "Importance",
    ascending=False
)

st.bar_chart(
    importance_data.set_index("Feature")
)

st.dataframe(
    importance_data,
    use_container_width=True
)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Dead Stock Early Warning System | Machine Learning Project"
)

