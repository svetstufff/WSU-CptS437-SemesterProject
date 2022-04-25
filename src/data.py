from sklearn.datasets import load_diabetes, make_regression
from sklearn.decomposition import PCA

# load diabetes dataset
X_diabetes, y_diabetes = load_diabetes(return_X_y=True)

# construct a regression dataset using sklearn's make_regression()
# this dataset can only have one one feature - this way, bias term and theta_1 can be plotted against loss function in 3D (see loss_paths.py)
X_one_feature, y_one_feature = make_regression(n_samples=500, n_features=1, noise=0.2, random_state=0)