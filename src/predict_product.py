import pandas as pd
import joblib

model = joblib.load("models/random_forest.pkl")

print("=" * 60)
print("DEAD STOCK EARLY WARNING SYSTEM")
print("=" * 60)

days_since_last_sale = float(input("Days since last sale: "))
sales_velocity = float(input("Sales velocity: "))
inventory_quantity = float(input("Inventory quantity: "))
product_age = float(input("Product age (days): "))
price = float(input("Price: "))
discount = float(input("Discount (%): "))

print("\nSeasonality:")
print("0 = Low")
print("1 = Medium")
print("2 = High")
seasonality = float(input("Enter seasonality: "))

customer_interest = float(input("Customer interest (0-100): "))

product = pd.DataFrame([{
    "Days_Since_Last_Sale": days_since_last_sale,
    "Sales_Velocity": sales_velocity,
    "Inventory_Quantity": inventory_quantity,
    "Product_Age": product_age,
    "Price": price,
    "Discount": discount,
    "Seasonality": seasonality,
    "Customer_Interest": customer_interest
}])

prediction = model.predict(product)[0]
probability = model.predict_proba(product)[0][1]

if probability < 0.30:
    risk = "LOW"
elif probability < 0.60:
    risk = "MEDIUM"
else:
    risk = "HIGH"

print("\n" + "=" * 60)
print("INVENTORY HEALTH RESULT")
print("=" * 60)

print(f"Dead Stock Probability: {probability * 100:.2f}%")
print(f"Risk Level: {risk}")

print("\nRISK FACTORS")
print("-" * 60)

factors = []

if days_since_last_sale > 60:
    factors.append("Product has not been sold for a long time.")

if sales_velocity < 2:
    factors.append("Sales velocity is very low.")

if inventory_quantity > 200:
    factors.append("Inventory quantity is very high.")

if product_age > 250:
    factors.append("Product has been in inventory for a long time.")

if customer_interest < 30:
    factors.append("Customer interest is low.")

if discount < 10 and probability > 0.5:
    factors.append("Discount may be insufficient to improve sales.")

if len(factors) == 0:
    print("No major risk factors detected.")
else:
    for factor in factors:
        print("• " + factor)

print("\nRECOMMENDATION")
print("-" * 60)

if risk == "HIGH":
    print("Consider increasing discount or running a promotional campaign.")
elif risk == "MEDIUM":
    print("Monitor this product closely and consider promotional activity.")
else:
    print("Continue normal inventory monitoring.")

print("=" * 60)