import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('IRIS.csv')

# Display the dataset and class counts
print(data.head())
print(data["species"].value_counts())

# Separating the independent variables from dependent variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

print("\nOriginal Data:")
print(data["species"].value_counts())

X_train['target'] = y_train
X_test['target'] = y_test

print("\nX_train:")
print(X_train["target"].value_counts())
print("\nX_test:")
print(X_test["target"].value_counts())

# 3D Scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot for training data
for species in y.unique():
    subset = X_train[X_train['target'] == species]
    ax.scatter(
        subset.iloc[:, 0],  # First feature
        subset.iloc[:, 1],  # Second feature
        subset.iloc[:, 2],  # Third feature
        label=species
    )

# Labels and legend
ax.set_title("3D Visualization of Iris Classification (Training Set)", fontsize=14)
ax.set_xlabel("Feature 1 (Sepal Length)")
ax.set_ylabel("Feature 2 (Sepal Width)")
ax.set_zlabel("Feature 3 (Petal Length)")
ax.legend(loc='best')

# Enable interactivity
plt.tight_layout()
plt.show()