from sklearn.datasets import load_breast_cancer
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
cancer = load_breast_cancer()

# Create DataFrame
df = pd.DataFrame(
    cancer.data,
    columns=cancer.feature_names
)

df["target"] = cancer.target

# Count the two classes
target_counts = df["target"].value_counts().sort_index()

print("Target counts:")
print(target_counts)

# Create bar chart
plt.bar(
    ["Malignant", "Benign"],
    target_counts
)

plt.title("Target Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Samples")

plt.show()

# Compare mean radius between the two classes
plt.figure(figsize=(8, 5))

df.boxplot(
    column="mean radius",
    by="target"
)

plt.title("Mean Radius by Diagnosis")
plt.suptitle("")
plt.xlabel("Diagnosis (0 = Malignant, 1 = Benign)")
plt.ylabel("Mean Radius")

plt.show()

import seaborn as sns

# Create correlation matrix
correlation_matrix = df.corr()

# Plot correlation heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    cmap="coolwarm",
    center=0
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.show()
