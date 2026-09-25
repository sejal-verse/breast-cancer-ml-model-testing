from sklearn.datasets import load_breast_cancer
import pandas as pd


# Load the dataset
cancer = load_breast_cancer()


# Convert the feature data into a DataFrame
df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)


# Add the target column
df["target"] = cancer.target


# Display basic information
print("Dataset loaded successfully!")
print()

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print()
print("First 5 rows:")
print(df.head())

print()
print("Target names:")
print(cancer.target_names)

print()
print("Target distribution:")
print(df["target"].value_counts())
