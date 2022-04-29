from sklearn.datasets import load_diabetes, fetch_california_housing, load_linnerud, make_regression
from sklearn.preprocessing import MinMaxScaler
import math

# preprocesses the provided dataset and returns the preprocessed instances
# currently, preprocessing is limited to normalizing feature values to avoid overflow erros
def preprocess(X, y):
    print(len(X))
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# load diabetes dataset
X_diabetes, y_diabetes = load_diabetes(return_X_y=True)

# load california housing dataset
X_cali, y_cali = preprocess(*fetch_california_housing(return_X_y=True))

X_house, y_house = load_linnerud(return_X_y=True)
y_house = list([round(targets[0]) for targets in y_house])

X_house, y_house = preprocess(X_house, y_house)

print(y_house)


# construct a regression dataset using sklearn's make_regression()
# this dataset can only have one one feature - this way, bias term and theta_1 can be plotted against loss function in 3D (see loss_paths.py)
X_one_feature, y_one_feature = make_regression(n_samples=500, n_features=1, noise=0.2, random_state=0)