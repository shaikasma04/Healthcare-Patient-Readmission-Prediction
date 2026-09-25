import pandas as pd
from pathlib import Path

# Locate the dataset
data_path = Path(__file__).resolve().parent.parent / "data" / "diabetic_data.csv"

# Load dataset
df = pd.read_csv(data_path)

print("========== DATA QUALITY ASSESSMENT ==========")

# 1. Dataset shape
print("\n1. Dataset Shape:")
print(df.shape)

# 2. Missing values
print("\n2. Missing Values:")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

print(missing)

# 3. Missing-value percentage
print("\n3. Missing Value Percentage:")
missing_percent = (missing / len(df)) * 100
print(missing_percent.round(2))

# 4. Duplicate records
print("\n4. Duplicate Records:")
print(df.duplicated().sum())

# 5. Target distribution
print("\n5. Readmission Distribution:")
print(df["readmitted"].value_counts())

# 6. Unique values in important categorical columns
print("\n6. Gender Values:")
print(df["gender"].value_counts(dropna=False))

print("\n7. Race Values:")
print(df["race"].value_counts(dropna=False))

print("\n8. Age Groups:")
print(df["age"].value_counts().sort_index())

# 7. Numerical summary
print("\n9. Numerical Summary:")
print(df.describe())

# 10. Check for '?' placeholder values
print("\n10. '?' Placeholder Values:")

question_mark_counts = {}

for column in df.columns:
    count = (df[column] == "?").sum()
    if count > 0:
        question_mark_counts[column] = count

question_mark_counts = pd.Series(question_mark_counts).sort_values(
    ascending=False
)

print(question_mark_counts)
# 11. Convert '?' placeholder values to missing values
print("\n11. Converting '?' to missing values:")

df.replace("?", pd.NA, inplace=True)

print("Conversion completed.")

print("\nMissing values after conversion:")
missing_after_conversion = df.isnull().sum()
missing_after_conversion = missing_after_conversion[
    missing_after_conversion > 0
].sort_values(ascending=False)

print(missing_after_conversion)

# 12. Missing-value summary after converting '?'
print("\n12. Missing-Value Summary:")

missing_summary = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (df.isnull().sum() / len(df)) * 100
})

missing_summary = missing_summary[
    missing_summary["Missing_Count"] > 0
].sort_values("Missing_Percentage", ascending=False)

print(missing_summary.round(2))
# 13. Check data types after missing-value conversion
print("\n13. Data Types:")

print(df.dtypes.value_counts())

print("\nCategorical Columns:")
print(df.select_dtypes(include=["str"]).columns.tolist())

print("\nNumerical Columns:")
print(df.select_dtypes(include=["number"]).columns.tolist())
# 14. Check unique values in categorical columns
print("\n14. Unique Values in Categorical Columns:")

categorical_columns = df.select_dtypes(include=["str"]).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(10))
# 15. Identify constant and near-constant categorical columns
print("\n15. Constant and Near-Constant Columns:")

categorical_columns = df.select_dtypes(include=["str"]).columns

for column in categorical_columns:
    value_counts = df[column].value_counts(dropna=False)

    if len(value_counts) == 1:
        print(f"{column}: CONSTANT - only one unique value")

    elif value_counts.iloc[0] / len(df) >= 0.99:
        percentage = (value_counts.iloc[0] / len(df)) * 100
        print(f"{column}: NEAR-CONSTANT - {percentage:.2f}% in one category")

# 16. Detect potential outliers using IQR
print("\n16. Potential Outliers in Numerical Columns:")

numerical_columns = df.select_dtypes(include=["number"]).columns

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ][column].count()

    print(
        f"{column}: {outliers} potential outliers "
        f"(Lower={lower_bound:.2f}, Upper={upper_bound:.2f})"
    )
# 17. Check for potentially invalid numerical values
print("\n17. Potentially Invalid Numerical Values:")

checks = {
    "time_in_hospital": df["time_in_hospital"] <= 0,
    "num_lab_procedures": df["num_lab_procedures"] < 0,
    "num_procedures": df["num_procedures"] < 0,
    "num_medications": df["num_medications"] < 0,
    "number_outpatient": df["number_outpatient"] < 0,
    "number_emergency": df["number_emergency"] < 0,
    "number_inpatient": df["number_inpatient"] < 0,
    "number_diagnoses": df["number_diagnoses"] <= 0
}

for column, condition in checks.items():
    print(f"{column}: {condition.sum()} potentially invalid values")


# 18. Check important identifier columns
print("\n18. Identifier Checks:")

print("Duplicate encounter_id:", df["encounter_id"].duplicated().sum())
print("Duplicate patient_nbr:", df["patient_nbr"].duplicated().sum())


# 19. Check target variable
print("\n19. Target Variable Check:")

print("Missing readmission values:", df["readmitted"].isna().sum())
print("\nReadmission categories:")
print(df["readmitted"].value_counts(dropna=False))


# 20. Check gender validity
print("\n20. Gender Validation:")

valid_gender = ["Male", "Female"]

invalid_gender = ~df["gender"].isin(valid_gender)

print("Potentially invalid gender values:", invalid_gender.sum())
print(df.loc[invalid_gender, "gender"].value_counts(dropna=False))


# 21. Check diabetes medication consistency
print("\n21. Diabetes Medication Check:")

print("diabetesMed values:")
print(df["diabetesMed"].value_counts(dropna=False))

print("\nchange values:")
print(df["change"].value_counts(dropna=False))


# 22. Final data-quality summary
print("\n22. FINAL DATA QUALITY SUMMARY:")

print("Total rows:", len(df))
print("Total columns:", len(df.columns))
print("Duplicate rows:", df.duplicated().sum())
print("Columns containing missing values:", df.isnull().any().sum())
print("Total missing cells:", df.isnull().sum().sum())

print("\nColumns with missing values:")
print(
    df.isnull().sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
)


# 23. Save cleaned assessment dataset
output_path = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "data_quality_assessment.csv"
)

df.to_csv(output_path, index=False)

print("\nData quality assessment file saved to:")
print(output_path)