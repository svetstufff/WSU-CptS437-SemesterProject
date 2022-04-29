from sklearn.datasets import load_diabetes, fetch_california_housing, load_linnerud, make_regression
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# preprocesses the provided dataset and returns the preprocessed instances
# currently, preprocessing is limited to normalizing feature values to avoid overflow erros
def preprocess(X, y):
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# load diabetes dataset
X_diabetes, y_diabetes = load_diabetes(return_X_y=True)

# load california housing dataset
X_cali, y_cali = preprocess(*fetch_california_housing(return_X_y=True))

# load linnerud dataset
X_linnerud, y_linnerud = load_linnerud(return_X_y=True)

# since this is a multiclass regression dataset, we only use the first class as the regression target
y_linnerud = np.array([targets[0] for targets in y_linnerud]) # only keep first class value
X_linnerud, y_linnerud = preprocess(X_linnerud, y_linnerud)


# construct a regression dataset using sklearn's make_regression()
# this dataset can only have one one feature - this way, bias term and theta_1 can be plotted against loss function in 3D (see loss_paths.py)
X_one_feature, y_one_feature = make_regression(n_samples=500, n_features=1, noise=0.2, random_state=0)