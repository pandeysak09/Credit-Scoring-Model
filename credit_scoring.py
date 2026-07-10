import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ==========================
# Load Processed Dataset
# ==========================

df = pd.read_csv("processed_credit_risk.csv")

# ==========================
# Features and Target
# ==========================

X = df.drop("loan_status", axis=1)

y = df["loan_status"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

# ==========================
# Feature Scaling
# ==========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==========================
# Logistic Regression
# ==========================

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================
# Predictions
# ==========================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# ==========================
# Evaluation Metrics
# ==========================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_prob)

print("\n==========================")
print("LOGISTIC REGRESSION RESULT")
print("==========================")

print("Accuracy :", round(accuracy,4))
print("Precision:", round(precision,4))
print("Recall   :", round(recall,4))
print("F1 Score :", round(f1,4))
print("ROC AUC  :", round(roc_auc,4))

# ==========================
# Classification Report
# ==========================

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# ==========================
# Confusion Matrix
# ==========================

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier

print("\n")
print("="*40)
print("RANDOM FOREST MODEL")
print("="*40)

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:,1]

print("Accuracy :", round(accuracy_score(y_test, rf_pred),4))
print("Precision:", round(precision_score(y_test, rf_pred),4))
print("Recall   :", round(recall_score(y_test, rf_pred),4))
print("F1 Score :", round(f1_score(y_test, rf_pred),4))
print("ROC AUC  :", round(roc_auc_score(y_test, rf_prob),4))

print("\nClassification Report:\n")
print(classification_report(y_test, rf_pred))
import matplotlib.pyplot as plt
import pandas as pd

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print(feature_importance)

plt.figure(figsize=(10,6))

plt.barh(
    feature_importance['Feature'],
    feature_importance['Importance']
)

plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, rf_prob)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()
import seaborn as sns

cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
import joblib

joblib.dump(rf, "credit_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model Saved Successfully")