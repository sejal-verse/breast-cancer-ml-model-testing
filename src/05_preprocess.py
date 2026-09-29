import importlib.util
from pathlib import Path

import pandas as pd


# Get the src folder path
src_folder = Path(__file__).parent

# Path to the dataset loader
loader_file = src_folder / "02_load_dataset.py"

# Load 02_load_dataset.py
spec = importlib.util.spec_from_file_location(
    "load_dataset_module",
    loader_file
)

load_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_dataset_module)

# Get the load_dataset function
load_dataset = load_dataset_module.load_dataset


def main():

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df, metadata = load_dataset()

    print("\n========== DATA PREPROCESSING ==========")

    print("\nOriginal dataset shape:")
    print(df.shape)

    # --------------------------------------------------
    # 2. Convert columns to numeric
    # --------------------------------------------------

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    print("\nData types after conversion:")
    print(df.dtypes)

    # --------------------------------------------------
    # 3. Check missing values
    # --------------------------------------------------

    print("\nMissing values:")
    print(df.isnull().sum().sum())

    # --------------------------------------------------
    # 4. Separate features and target
    # --------------------------------------------------

    X = df.drop("Result", axis=1)
    y = df["Result"]

    print("\nFeatures (X) shape:")
    print(X.shape)

    print("\nTarget (y) shape:")
    print(y.shape)

    # --------------------------------------------------
    # 5. Display target distribution
    # --------------------------------------------------

    print("\nTarget distribution:")
    print(y.value_counts())

    # --------------------------------------------------
    # 6. Display feature names
    # --------------------------------------------------

    print("\nNumber of features:")
    print(X.shape[1])

    print("\nFeature names:")
    print(X.columns.tolist())

    print("\n========== PREPROCESSING COMPLETE ==========")


if __name__ == "__main__":
    main()
    