import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load cleaned dataset
df = pd.read_csv("cleaned_churn.csv")

print("Dataset loaded successfully!")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())
# Separate Features (X) and Target (y)

X = df.drop(columns=["customerID", "Churn"])

y = df["Churn"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Values:")
print(y.value_counts())
# Identify numeric and categorical columns

numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns

categorical_columns = X.select_dtypes(include=["object", "str"]).columns

print("\nNumeric Columns:")
print(list(numeric_columns))

print("\nCategorical Columns:")
print(list(categorical_columns))
# Create preprocessing transformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

print("\nPreprocessor created successfully!")
# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nTrain-Test Split Successful!")
# Create Logistic Regression model

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
))
])

# Train the model

model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")
# Make predictions

y_pred = model.predict(X_test)
# Predict churn probabilities
y_probability = model.predict_proba(X_test)[:, 1]

# Calculate ROC-AUC Score
roc_auc = roc_auc_score(
    (y_test == "Yes").astype(int),
    y_probability
)

print("\nROC-AUC Score:", round(roc_auc, 4))

# Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")
# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save trained model
# Feature Importance using Logistic Regression

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

coefficients = model.named_steps["classifier"].coef_[0]

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": coefficients
})

feature_importance["Absolute_Importance"] = (
    feature_importance["Importance"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Absolute_Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
joblib.dump(model, "churn_model.pkl")

print("\nModel saved successfully!")
