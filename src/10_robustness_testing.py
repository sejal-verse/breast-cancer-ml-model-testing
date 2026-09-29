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
    print("\n========== ROBUSTNESS TESTING ==========")

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

    # ---------- ORIGINAL PREDICTIONS ----------
    original_predictions = model.predict(X_test)

    # ---------- CREATE PERTURBED DATA ----------
    X_perturbed = X_test.copy()

    # Change one feature slightly.
    # We use the first feature only so that the experiment
    # remains controlled and easy to interpret.
    feature_to_change = X_perturbed.columns[0]

    print(f"\nFeature being tested: {feature_to_change}")

    # Flip the value between -1 and +1.
    # This creates a controlled feature perturbation.
    X_perturbed[feature_to_change] = (
        X_perturbed[feature_to_change] * -1
    )

    # ---------- PERTURBED PREDICTIONS ----------
    perturbed_predictions = model.predict(X_perturbed)

    # ---------- COMPARE PREDICTIONS ----------
    results = X_test.copy()

    results["Original_Prediction"] = original_predictions
    results["Perturbed_Prediction"] = perturbed_predictions

    results["Prediction_Changed"] = (
        results["Original_Prediction"]
        != results["Perturbed_Prediction"]
    )

    results["Actual"] = y_test

    # Store original and modified feature values
    results["Original_Feature_Value"] = X_test[feature_to_change]
    results["Perturbed_Feature_Value"] = X_perturbed[feature_to_change]

    # ---------- SUMMARY ----------
    total_samples = len(results)
    changed_predictions = results["Prediction_Changed"].sum()
    unchanged_predictions = total_samples - changed_predictions

    stability_rate = unchanged_predictions / total_samples
    change_rate = changed_predictions / total_samples

    print("\n========== ROBUSTNESS RESULTS ==========")
    print(f"Total test samples       : {total_samples}")
    print(f"Predictions unchanged   : {unchanged_predictions}")
    print(f"Predictions changed     : {changed_predictions}")
    print(f"Stability rate          : {stability_rate:.4f}")
    print(f"Prediction change rate  : {change_rate:.4f}")

    # ---------- SAVE RESULTS ----------
    reports_folder = src_folder.parent / "reports"
    reports_folder.mkdir(exist_ok=True)

    output_file = reports_folder / "robustness_results.csv"

    results.to_csv(output_file, index=True)

    print("\nResults saved to:")
    print(output_file)

    print("\nRobustness testing completed successfully!")


if __name__ == "__main__":
    main()
    