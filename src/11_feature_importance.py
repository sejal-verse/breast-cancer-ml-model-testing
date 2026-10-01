import importlib.util
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ---------- LOAD DATASET ----------
src_folder = Path(__file__).parent
loader_file = src_folder / "02_load_dataset.py"

spec = importlib.util.spec_from_file_location(
    "load_dataset_module",
    loader_file
)

load_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_dataset_module)

load_dataset = load_dataset_module.load_dataset


def main():
    print("\n========== FEATURE IMPORTANCE ANALYSIS ==========")

    # Load dataset
    df, metadata = load_dataset()

    # Convert columns to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Separate features and target
    X = df.drop("Result", axis=1)
    y = df["Result"]

    # Same split used in previous steps
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # ---------- FEATURE IMPORTANCE ----------
    importance_values = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance_values
    })

    # Sort from most important to least important
    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    # Reset index
    importance_df = importance_df.reset_index(drop=True)

    print("\n========== FEATURE IMPORTANCE ==========")
    print(
        importance_df.to_string(
            index=False,
            formatters={
                "Importance": "{:.6f}".format
            }
        )
    )

    # ---------- SAVE CSV ----------
    reports_folder = src_folder.parent / "reports"
    figures_folder = reports_folder / "figures"

    reports_folder.mkdir(exist_ok=True)
    figures_folder.mkdir(exist_ok=True)

    csv_file = reports_folder / "feature_importance.csv"

    importance_df.to_csv(
        csv_file,
        index=False
    )

    # ---------- CREATE VISUALIZATION ----------
    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Decision Tree Feature Importance")

    # Most important feature appears at the top
    plt.gca().invert_yaxis()

    plt.tight_layout()

    figure_file = figures_folder / "feature_importance.png"

    plt.savefig(
        figure_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # ---------- COMPLETION ----------
    print("\n========== FILES CREATED ==========")
    print(f"CSV       : {csv_file}")
    print(f"Visualization: {figure_file}")

    print("\nFeature importance analysis completed successfully!")


if __name__ == "__main__":
    main()
    