import importlib.util
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------
# Load dataset function from 02_load_dataset.py
# --------------------------------------------------

src_folder = Path(__file__).parent
loader_file = src_folder / "02_load_dataset.py"

spec = importlib.util.spec_from_file_location(
    "load_dataset_module",
    loader_file
)

load_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(load_dataset_module)

load_dataset = load_dataset_module.load_dataset


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n========== SAVE FINAL MODEL ==========")

    # Load dataset
    df, metadata = load_dataset()

    # Convert columns to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Separate features and target
    X = df.drop("Result", axis=1)
    y = df["Result"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------
    # Final model
    # --------------------------------------------------

    model = DecisionTreeClassifier(
        max_depth=7,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    print("\nTraining final model...")

    model.fit(X_train, y_train)

    print("Model training completed.")

    # --------------------------------------------------
    # Create models directory
    # --------------------------------------------------

    project_folder = src_folder.parent
    models_folder = project_folder / "models"

    models_folder.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_file = models_folder / "phishing_model.pkl"

    joblib.dump(model, model_file)

    # --------------------------------------------------
    # Save feature columns
    # --------------------------------------------------

    feature_file = models_folder / "feature_columns.pkl"

    joblib.dump(list(X.columns), feature_file)

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print("\n========== FILES CREATED ==========")

    print(f"Model: {model_file}")
    print(f"Features: {feature_file}")

    print("\nNumber of features:", len(X.columns))

    print("\nFinal model saved successfully!")


if __name__ == "__main__":
    main()
    