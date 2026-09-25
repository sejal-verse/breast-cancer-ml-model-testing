from sklearn.datasets import load_breast_cancer
import pandas as pd


# Load dataset
cancer = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

# Add target
df["target"] = cancer.target

# 1. Basic dataset information

print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)

print(df.shape)

# 2. Feature names

print("\n" + "=" * 50)
print("FEATURE NAMES")
print("=" * 50)

for feature in cancer.feature_names:
    print(feature)

# 3. Data types

print("\n" + "=" * 50)
print("DATA TYPES")
print("=" * 50)

print(df.dtypes)

# 4. Missing values

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)

print(df.isnull().sum().sum())

# 5. Target distribution

print("\n" + "=" * 50)
print("TARGET DISTRIBUTION")
print("=" * 50)

print(df["target"].value_counts())

print("\nTarget names:")
print(cancer.target_names)

# 6. Statistical summary

print("\n" + "=" * 50)
print("STATISTICAL SUMMARY")
print("=" * 50)

print(df.describe())
