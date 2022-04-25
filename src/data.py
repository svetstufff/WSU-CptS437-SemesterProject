from sklearn.datasets import load_diabetes

# load diabetes dataset
X, y = load_diabetes(return_X_y=True)

# use only a subset of dataset to control runtime
X, y = X[:1000], y[:1000]