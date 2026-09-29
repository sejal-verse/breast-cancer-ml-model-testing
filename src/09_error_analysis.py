import importlib.util
from pathlib import Path

import pandas as pd
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
    print("\n========== ERROR ANALYSIS ==========")

    # Load dataset
    df, metadata = load_dataset()

    # Convert all columns to numeric
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Separate features and target
    X = df.drop("Result", axis=1)
    y = df["Result"]

    # Same split as Step 07
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train the same Decision Tree model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Create analysis dataframe
    error_df = X_test.copy()

    error_df["Actual"] = y_test
    error_df["Predicted"] = y_pred

    # Identify incorrect predictions
    error_df["Correct"] = error_df["Actual"] == error_df["Predicted"]

    errors = error_df[error_df["Correct"] == False].copy()

    # Classify errors
    def classify_error(row):
        actual = row["Actual"]
        predicted = row["Predicted"]

        # Phishing = -1
        # Legitimate = +1

        if actual == 1 and predicted == -1:
            return "False Positive (Legitimate -> Phishing)"

        elif actual == -1 and predicted == 1:
            return "False Negative (Phishing -> Legitimate)"

        return "Correct"

    errors["Error_Type"] = errors.apply(classify_error, axis=1)

    # ---------- SUMMARY ----------
    total_samples = len(error_df)
    total_errors = len(errors)

    false_positives = len(
        errors[errors["Error_Type"] == "False Positive (Legitimate -> Phishing)"]
    )

    false_negatives = len(
        errors[errors["Error_Type"] == "False Negative (Phishing -> Legitimate)"]
    )

    error_rate = total_errors / total_samples

    print("\n========== ERROR SUMMARY ==========")
    print(f"Total test samples : {total_samples}")
    print(f"Total errors       : {total_errors}")
    print(f"Error rate         : {error_rate:.4f}")
    print(f"False positives    : {false_positives}")
    print(f"False negatives    : {false_negatives}")

    # ---------- SAVE RESULTS ----------
    reports_folder = src_folder.parent / "reports"
    reports_folder.mkdir(exist_ok=True)

    output_file = reports_folder / "error_analysis.csv"

    errors.to_csv(output_file, index=True)

    print("\n========== ERROR ANALYSIS ==========")
    print(f"Misclassified samples saved to:")
    print(output_file)

    print("\nError analysis completed successfully!")


if __name__ == "__main__":
    main()
    