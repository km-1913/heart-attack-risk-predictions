import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Student Mental health.csv")

# Remove Timestamp column
if "Timestamp" in df.columns:
    df.drop("Timestamp", axis=1, inplace=True)

# Remove missing values
df.dropna(inplace=True)

# -----------------------------
# Encode categorical columns
# -----------------------------
encoders = {}

for column in df.columns:
    if df[column].dtype == "object":
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        encoders[column] = le

# -----------------------------
# Target variable
# -----------------------------
target_column = "Do you have Depression?"

X = df.drop(target_column, axis=1)
y = df[target_column]

# Save feature names for Streamlit
feature_names = X.columns.tolist()

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# -----------------------------
# Save files for website
# -----------------------------
joblib.dump(model, "mental_health_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("Files saved successfully!")