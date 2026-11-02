import pandas as pd

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first 5 rows
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)
# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())
# Check rows where Churn is missing
print("\nRows with missing Churn:")
print(df[df["Churn"].isnull()].head(10))

# Count missing Churn rows
print("\nTotal missing Churn rows:")
print(df["Churn"].isnull().sum())
# Check records after original 7043 customers

print("\nLast 10 rows of dataset:")
print(df.tail(10))

print("\nDataset shape:")
print(df.shape)
# Check valid customer records

valid_customers = df[df["Churn"].notna()]

print("\nTotal valid customer records:")
print(len(valid_customers))

print("\nChurn values:")
print(valid_customers["Churn"].value_counts())
# Keep only valid customer records

clean_df = df[df["Churn"].notna()].copy()

# Check cleaned dataset
print("\nOriginal Dataset Shape:")
print(df.shape)

print("\nCleaned Dataset Shape:")
print(clean_df.shape)

# Save cleaned dataset
clean_df.to_csv("cleaned_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")
# Check duplicate customers

print("\nDuplicate customer IDs:")
print(clean_df["customerID"].duplicated().sum())

print("\nTotal duplicate rows:")
print(clean_df.duplicated().sum())
# Check data types

print("\nColumn Data Types:")
print(clean_df.dtypes)
# Convert TotalCharges into numeric format

clean_df["TotalCharges"] = pd.to_numeric(
    clean_df["TotalCharges"],
    errors="coerce"
)

print("\nTotalCharges Data Type:")
print(clean_df["TotalCharges"].dtype)

print("\nMissing TotalCharges:")
print(clean_df["TotalCharges"].isnull().sum())
# Check customers with missing TotalCharges

print("\nCustomers with missing TotalCharges:")
print(
    clean_df.loc[
        clean_df["TotalCharges"].isnull(),
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges"]
    ]
)
# Fill missing TotalCharges with 0
clean_df["TotalCharges"] = clean_df["TotalCharges"].fillna(0)

# Check missing values again
print("\nMissing TotalCharges after cleaning:")
print(clean_df["TotalCharges"].isnull().sum())

# Check data type
print("\nTotalCharges data type:")
print(clean_df["TotalCharges"].dtype)
# Check all remaining missing values

print("\nRemaining Missing Values:")
print(clean_df.isnull().sum())

print("\nTotal Missing Values:")
print(clean_df.isnull().sum().sum())
# Save final cleaned dataset

clean_df.to_csv("cleaned_churn.csv", index=False)

print("\nFinal cleaned dataset saved!")
print("Final Shape:", clean_df.shape)
# Correct data types

clean_df["tenure"] = pd.to_numeric(clean_df["tenure"])

clean_df["SeniorCitizen"] = clean_df["SeniorCitizen"].astype(int)

print("\nUpdated Data Types:")
print(clean_df.dtypes[["tenure", "SeniorCitizen"]])
# Save final cleaned dataset again

clean_df.to_csv("cleaned_churn.csv", index=False)

print("\nFinal dataset saved successfully!")
print("Final Shape:", clean_df.shape)

print("\nFinal Data Types:")
print(clean_df[["tenure", "SeniorCitizen", "TotalCharges"]].dtypes)