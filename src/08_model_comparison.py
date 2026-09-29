import importlib.util
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# Get src folder
src_folder = Path(__file__).parent

# Load dataset loader
loader_file = src_folder / "02_load_dataset.py"

spec = importlib.util.spec_from_file_location(
    "load_dataset_module",
    loader_file
)

load_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_dataset_module)

load_dataset = load_dataset_module.load_dataset


def main():

    print("\n========== MODEL COMPARISON ==========")

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df, metadata = load_dataset()

    # Convert values to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Separate features and target
    X = df.drop("Result", axis=1)
    y = df["Result"]

    # --------------------------------------------------
    # 2. Train/Test Split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # 3. Define models
    # --------------------------------------------------

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        )
    }

    results = []

    # --------------------------------------------------
    # 4. Train and evaluate each model
    # --------------------------------------------------

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=1
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=1
        )

        f1 = f1_score(
            y_test,
            y_pred,
            pos_label=1
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # --------------------------------------------------
    # 5. Display comparison
    # --------------------------------------------------

    results_df = pd.DataFrame(results)

    print("\n========== MODEL COMPARISON RESULTS ==========")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format
            }
        )
    )

    print("\nModel comparison completed successfully!")


if __name__ == "__main__":
    main()
    