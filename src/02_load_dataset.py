from pathlib import Path
from scipy.io import arff
import pandas as pd


DATA_PATH = Path("data/raw/phishing.arff")


def load_dataset():
    data, metadata = arff.loadarff(DATA_PATH)

    df = pd.DataFrame(data)

    # ARFF categorical values may be stored as bytes.
    # Convert them to normal strings.
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].apply(
            lambda value: value.decode("utf-8")
            if isinstance(value, bytes)
            else value
        )

    return df, metadata


if __name__ == "__main__":
    df, metadata = load_dataset()

    print("========== DATASET LOADED ==========")

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nDataset loaded successfully!")
    