import sys

print("========== ENVIRONMENT TEST ==========")

print("Python version:")
print(sys.version)

print("\nPython executable:")
print(sys.executable)

print("\nTesting libraries...")

import numpy
import pandas
import sklearn
import xgboost
import matplotlib
import seaborn
import scipy

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", seaborn.__version__)
print("SciPy:", scipy.__version__)

print("\nEnvironment test successful!")
