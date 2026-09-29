import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
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

    # Load dataset
    df, metadata = load_dataset()

    print("\n========== DATA VISUALIZATION ==========")

    print("\nDataset loaded successfully!")
    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    # Show value distributions
    for column in df.columns:
        print(f"\n--- {column} ---")
        print(df[column].value_counts())


if __name__ == "__main__":
    main()
    
