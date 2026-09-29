# 🛍️ Dead Stock Early Warning System

## 📌 Project Overview

The Dead Stock Early Warning System is a Machine Learning based inventory management system that helps retailers identify products that are at risk of becoming dead stock.

The system analyzes product-level information such as:

* Days since last sale
* Sales velocity
* Inventory quantity
* Product age
* Price
* Discount
* Seasonality
* Customer interest

It predicts the probability that a product may become dead stock and assigns a risk level:

* LOW
* MEDIUM
* HIGH

The system also uses K-Means clustering to segment products based on their inventory and sales characteristics.

---

## 🎯 Problem Statement

Retailers may hold large quantities of products that sell slowly or stop selling completely.

This can result in:

* Excess inventory
* Blocked working capital
* Storage costs
* Discounting pressure
* Reduced inventory efficiency

The proposed system provides an early warning mechanism to identify potentially problematic products.

---

## 💡 Proposed Solution

The system combines Machine Learning classification and clustering techniques.

### Classification

Three models are trained and compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The Random Forest model is used by the prediction system.

### Clustering

K-Means clustering is used to group products into categories such as:

* Fast Sellers
* Slow Sellers
* Seasonal Products
* Dead-Stock Risk

---

## 🧠 Machine Learning Workflow

```text
Synthetic Dataset
       ↓
Data Inspection
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Train/Test Split
       ↓
Classification Models
       ↓
Model Comparison
       ↓
Random Forest Prediction
       ↓
K-Means Clustering
       ↓
Streamlit Dashboard
```

---

## 📊 Dataset

The project uses a synthetically generated dataset containing 5,000 products.

Each product contains information about:

| Feature              | Description                                      |
| -------------------- | ------------------------------------------------ |
| Product_ID           | Unique product identifier                        |
| Product_Name         | Product name                                     |
| Category             | Product category                                 |
| Days_Since_Last_Sale | Number of days since the product was last sold   |
| Sales_Velocity       | Rate at which the product sells                  |
| Inventory_Quantity   | Current quantity in inventory                    |
| Product_Age          | Number of days the product has been in inventory |
| Price                | Product price                                    |
| Discount             | Current discount percentage                      |
| Seasonality          | Seasonal demand indicator                        |
| Customer_Interest    | Customer interest score                          |
| Dead_Stock           | Target variable                                  |

---

## 🤖 Models Used

### Logistic Regression

Used as a classification model to predict whether a product is likely to become dead stock.

### Decision Tree

Uses decision rules based on product characteristics to classify products.

### Random Forest

Combines multiple decision trees to produce the final classification used by the prediction system.

### K-Means

Groups products with similar characteristics into inventory segments.

---

## 📈 Evaluation Metrics

The classification models are evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The project also provides a comparison of the three classification models through the Streamlit dashboard.

---

## 🖥️ Dashboard

The project includes a Streamlit dashboard with:

### Inventory Overview

Displays:

* Total products
* Dead stock products
* Average inventory
* Average sales velocity

### Product Prediction

Users can enter product information and obtain:

* Dead stock probability
* Risk level
* Risk factors
* Recommended action

### Product Search

Users can search for products using:

* Product ID
* Product name

### Product Segmentation

Displays K-Means based inventory segments.

### Model Comparison

Displays the performance of the classification models.

### Feature Importance

Displays the importance of different product features according to the Random Forest model.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## 📁 Project Structure

```text
Dead-Stock-Early-Warning-System/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   │   └── dead_stock_dataset.csv
│   │
│   └── processed/
│       ├── graphs/
│       ├── train_data.csv
│       ├── test_data.csv
│       ├── model_comparison.csv
│       ├── clustered_products.csv
│       └── clustered_products_labeled.csv
│
├── models/
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   └── random_forest.pkl
│
├── notebooks/
│
├── src/
│   ├── data_generation.py
│   ├── inspect_data.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── train_logistic_regression.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── compare_models.py
│   ├── clustering.py
│   ├── label_clusters.py
│   └── predict_product.py
│
├── requirements.txt
├── README.md
└── venv/
```

---

## ▶️ How to Run

### 1. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Run the dashboard

```powershell
python -m streamlit run app/dashboard.py
```

The Streamlit application will open in the browser.

---

## ⚠️ Dataset Limitation

This prototype uses synthetically generated data.

The `Dead_Stock` target was generated using a predefined risk-scoring rule based on the product features. Therefore, the model results demonstrate the machine-learning workflow and prototype functionality but should not be interpreted as evidence of real-world predictive performance.

A production version would require historical retail sales and inventory data with real future dead-stock outcomes.

---

## 🚀 Future Improvements

Possible future extensions include:

* Real retailer inventory data
* Time-series sales forecasting
* Automated discount recommendations
* Product-level explainable AI
* Real-time inventory database
* Automated alerts for high-risk products
* Seasonal demand forecasting
* Integration with e-commerce platforms
* Cloud deployment
* Role-based retailer dashboards

---

## 👩‍💻 Project

**Dead Stock Early Warning System**

Machine Learning Project
