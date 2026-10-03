import importlib.util

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ==================================================
# LOAD DATASET
# ==================================================

src_folder = Path(__file__).parent

loader_file = src_folder / "02_load_dataset.py"

spec = importlib.util.spec_from_file_location(
    "load_dataset_module",
    loader_file
)

load_dataset_module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(load_dataset_module)

load_dataset = load_dataset_module.load_dataset


# ==================================================
# CALCULATE METRICS
# ==================================================

def calculate_metrics(model, X_test, y_test):
    """
    Calculate final evaluation metrics.

    Phishing (-1) is treated as the positive class.
    """

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------

    predictions = model.predict(X_test)


    # --------------------------------------------------
    # Probability Predictions
    # --------------------------------------------------

    probabilities = model.predict_proba(X_test)

    # Find the column corresponding to phishing (-1)
    phishing_class_index = list(model.classes_).index(-1)

    phishing_probabilities = probabilities[:, phishing_class_index]


    # --------------------------------------------------
    # Convert Target for ROC-AUC
    # --------------------------------------------------

    # Original dataset:
    # -1 = Phishing
    # +1 = Legitimate
    #
    # ROC-AUC expects the positive class to be represented
    # as 1, so we explicitly convert:
    #
    # Phishing (-1)    -> 1
    # Legitimate (+1) -> 0

    y_test_binary = (y_test == -1).astype(int)


    # --------------------------------------------------
    # Calculate Metrics
    # --------------------------------------------------

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            pos_label=-1
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            pos_label=-1
        ),

        "F1 Score": f1_score(
            y_test,
            predictions,
            pos_label=-1
        ),

        "ROC-AUC": roc_auc_score(
            y_test_binary,
            phishing_probabilities
        )
    }

    return metrics, predictions


# ==================================================
# MAIN
# ==================================================

def main():

    print("\n========== FINAL MODEL EVALUATION ==========")


    # ==================================================
    # LOAD DATA
    # ==================================================

    df, metadata = load_dataset()


    # Convert all columns to numeric values
    for column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # Separate features and target

    X = df.drop(
        "Result",
        axis=1
    )

    y = df["Result"]


    # ==================================================
    # TRAIN / TEST SPLIT
    # ==================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.2,

        random_state=42,

        stratify=y
    )


    # ==================================================
    # BASELINE MODEL
    # ==================================================

    print("\nTraining baseline Decision Tree...")

    baseline_model = DecisionTreeClassifier(
        random_state=42
    )

    baseline_model.fit(
        X_train,
        y_train
    )


    baseline_metrics, _ = calculate_metrics(
        baseline_model,
        X_test,
        y_test
    )


    # ==================================================
    # IMPROVED MODEL
    # ==================================================

    print("Training improved Decision Tree...")


    # These parameters come from the model-improvement
    # stage of the project.

    improved_model = DecisionTreeClassifier(

        max_depth=10,

        min_samples_split=2,

        min_samples_leaf=1,

        random_state=42
    )


    improved_model.fit(
        X_train,
        y_train
    )


    improved_metrics, improved_predictions = calculate_metrics(

        improved_model,

        X_test,

        y_test
    )


    # ==================================================
    # RESULTS TABLE
    # ==================================================

    results = pd.DataFrame([

        {
            "Model": "Baseline Decision Tree",
            **baseline_metrics
        },

        {
            "Model": "Improved Decision Tree",
            **improved_metrics
        }

    ])


    # ==================================================
    # PRINT RESULTS
    # ==================================================

    print("\n========== FINAL RESULTS ==========")

    print(

        results.to_string(

            index=False,

            formatters={

                "Accuracy": "{:.4f}".format,

                "Precision": "{:.4f}".format,

                "Recall": "{:.4f}".format,

                "F1 Score": "{:.4f}".format,

                "ROC-AUC": "{:.4f}".format
            }
        )
    )


    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    cm = confusion_matrix(

        y_test,

        improved_predictions,

        labels=[-1, 1]
    )


    print(
        "\n========== IMPROVED MODEL CONFUSION MATRIX =========="
    )

    print(cm)


    # ==================================================
    # CREATE REPORT FOLDERS
    # ==================================================

    reports_folder = src_folder.parent / "reports"

    figures_folder = reports_folder / "figures"

    reports_folder.mkdir(
        exist_ok=True
    )

    figures_folder.mkdir(
        exist_ok=True
    )


    # ==================================================
    # SAVE RESULTS
    # ==================================================

    results_file = reports_folder / "final_results.csv"

    results.to_csv(

        results_file,

        index=False
    )


    # ==================================================
    # CREATE CONFUSION MATRIX FIGURE
    # ==================================================

    display = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=[
            "Phishing (-1)",
            "Legitimate (+1)"
        ]
    )


    display.plot()

    plt.title(
        "Improved Decision Tree - Confusion Matrix"
    )

    plt.tight_layout()


    figure_file = (
        figures_folder /
        "final_confusion_matrix.png"
    )


    plt.savefig(

        figure_file,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close()


    # ==================================================
    # COMPLETION
    # ==================================================

    print("\n========== FILES CREATED ==========")

    print(
        f"Results: {results_file}"
    )

    print(
        f"Confusion Matrix: {figure_file}"
    )

    print(
        "\nFinal model evaluation completed successfully!"
    )


# ==================================================
# RUN PROGRAM
# ==================================================

if __name__ == "__main__":

    main()
    