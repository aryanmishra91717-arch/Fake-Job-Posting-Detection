import pandas as pd

df = pd.read_csv("fake_job_postings.csv")

print("Dataset loaded successfully!")
print("Number of rows and columns:", df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 job postings:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

print("\nFraudulent job postings:")
print(df["fraudulent"].value_counts())
print("\nData types:")
print(df.dtypes)
print("\nFraudulent job distribution:")
print(df["fraudulent"].value_counts())

print("\nFraudulent job percentage:")
print(df["fraudulent"].value_counts(normalize=True) * 100)
print("\nCleaning data...")

# Fill missing text values
text_columns = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

for col in text_columns:
    df[col] = df[col].fillna("")

print("Missing values after cleaning:")
print(df.isnull().sum())
text_columns = [
    "title",
    "location",
    "department",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]
for col in text_columns:
    df[col] = df[col].fillna("Unknown")
    df["salary_range"] = df["salary_range"].fillna("Unknown")
    print("\nMissing values after cleaning:")
print(df.isnull().sum())
print("\nCleaning completed successfully!")
print("Final dataset shape:", df.shape)
df.to_csv("cleaned_job_postings.csv", index=False)
print("Cleaned dataset saved successfully!")