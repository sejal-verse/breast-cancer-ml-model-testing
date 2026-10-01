import importlib.util
from pathlib import Path

import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


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


def evaluate_model(model, X_test, y_test):
    """Calculate evaluation metrics for a model."""

    predictions = model.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test,
            predictions,
            pos_label=1
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            pos_label=1
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            pos_label=1
        )
    }


def main():
    print("\n========== MODEL IMPROVEMENT ==========")

    # ---------- LOAD DATA ----------
    df, metadata = load_dataset()

    for column in df.columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    X = df.drop("Result", axis=1)
    y = df["Result"]

    # ---------- TRAIN / TEST SPLIT ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ---------- BASELINE MODEL ----------
    baseline_model = DecisionTreeClassifier(
        random_state=42
    )

    baseline_model.fit(
        X_train,
        y_train
    )

    baseline_results = evaluate_model(
        baseline_model,
        X_test,
        y_test
    )

    # ---------- HYPERPARAMETER SEARCH ----------
    print("\nSearching for better Decision Tree parameters...")

    parameter_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(
            random_state=42
        ),
        param_grid=parameter_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    # ---------- BEST MODEL ----------
    improved_model = grid_search.best_estimator_

    improved_results = evaluate_model(
        improved_model,
        X_test,
        y_test
    )

    # ---------- DISPLAY BEST PARAMETERS ----------
    print("\n========== BEST PARAMETERS ==========")

    for parameter, value in grid_search.best_params_.items():
        print(f"{parameter}: {value}")

    print(
        f"\nBest cross-validation F1 score: "
        f"{grid_search.best_score_:.4f}"
    )

    # ---------- COMPARISON ----------
    comparison = pd.DataFrame([
        {
            "Model": "Baseline Decision Tree",
            **baseline_results
        },
        {
            "Model": "Tuned Decision Tree",
            **improved_results
        }
    ])

    print("\n========== MODEL COMPARISON ==========")

    print(
        comparison.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format
            }
        )
    )

    # ---------- SAVE RESULTS ----------
    reports_folder = src_folder.parent / "reports"
    reports_folder.mkdir(exist_ok=True)

    output_file = (
        reports_folder /
        "model_improvement.csv"
    )

    comparison.to_csv(
        output_file,
        index=False
    )

    print("\nResults saved to:")
    print(output_file)

    print("\nModel improvement completed successfully!")


if __name__ == "__main__":
    main()
    