import pandas as pd
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load data
train_data = pd.read_csv("data/processed/train_data.csv")
test_data = pd.read_csv("data/processed/test_data.csv")

# Separate features and target
X_train = train_data.drop("Dead_Stock", axis=1)
y_train = train_data["Dead_Stock"]

X_test = test_data.drop("Dead_Stock", axis=1)
y_test = test_data["Dead_Stock"]

# Create Decision Tree
model = DecisionTreeClassifier(
    max_depth=6,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("=" * 50)
print("DECISION TREE RESULTS")
print("=" * 50)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(
    model,
    "models/decision_tree.pkl"
)

print("\nModel saved successfully!")
print("models/decision_tree.pkl")