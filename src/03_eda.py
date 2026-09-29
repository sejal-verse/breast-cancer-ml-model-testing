import pandas as pd
import importlib.util
from pathlib import Path


# Get the src folder path
src_folder = Path(__file__).parent

# Path to the dataset loader
loader_file = src_folder / "02_load_dataset.py"

# Load the 02_load_dataset.py file
spec = importlib.util.spec_from_file_location(
    "load_dataset_module",
    loader_file
)

load_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_dataset_module)

# Get the load_dataset function
load_dataset = load_dataset_module.load_dataset


def main():

    # Load dataset
    df, metadata = load_dataset()

    print("\n--- Dataset Shape ---")
    print(df.shape)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Names ---")
    print(df.columns.tolist())

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Dataset Information ---")
    df.info()


if __name__ == "__main__":
    main()
    