import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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

    print("\n========== MODEL EVALUATION ==========")

    # Load dataset
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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Create model
    model = DecisionTreeClassifier(
        random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    print("\n========== RESULTS ==========")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    # --------------------------------------------------
    # Classification Report
    # --------------------------------------------------

    print("\n========== CLASSIFICATION REPORT ==========")
    print(classification_report(y_test, y_pred))

    print("\nModel evaluation completed successfully!")


if __name__ == "__main__":
    main()
    
    