import importlib.util
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Get the src folder path
src_folder = Path(__file__).parent

# Path to the preprocessing file
preprocess_file = src_folder / "05_preprocess.py"

# Load 05_preprocess.py
spec = importlib.util.spec_from_file_location(
    "preprocess_module",
    preprocess_file
)

preprocess_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess_module)


def main():

    print("\n========== BASELINE MODEL ==========")

    # --------------------------------------------------
    # 1. Load and preprocess dataset
    # --------------------------------------------------

    df, metadata = preprocess_module.load_dataset()

    # Convert columns to numeric
    for column in df.columns:
        df[column] = __import__("pandas").to_numeric(
            df[column],
            errors="coerce"
        )

    # Separate features and target
    X = df.drop("Result", axis=1)
    y = df["Result"]

    print("\nDataset loaded.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # --------------------------------------------------
    # 2. Split dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain/Test Split:")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    # --------------------------------------------------
    # 3. Create baseline model
    # --------------------------------------------------

    model = DecisionTreeClassifier(
        random_state=42
    )

    # --------------------------------------------------
    # 4. Train model
    # --------------------------------------------------

    model.fit(X_train, y_train)

    print("\nModel trained successfully!")

    # --------------------------------------------------
    # 5. Make predictions
    # --------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------
    # 6. Calculate accuracy
    # --------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)

    print("\n========== BASELINE RESULTS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nBaseline model completed successfully!")


if __name__ == "__main__":
    main()
    
    