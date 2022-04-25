from sklearn.datasets import load_diabetes

# load diabetes dataset
X, y = load_diabetes(return_X_y=True)

# use only a subset of dataset to control runtime
X, y = X[:100], y[:100]