from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression
from sklearn.linear_model import LinearRegression as LR

# load diabetes dataset and define classifiers
l = LR()
X, y = load_diabetes(return_X_y=True)
X, y = X[:1000], y[:1000]
alpha = 0.0052
iterations=100
classifiers = {
    "linear regression": LinearRegression(alpha=alpha, iterations=iterations),
    "greedy": GreedyLinearRegression(alpha=alpha, iterations=iterations) 
}

# plot the learning curve for simple linear regression and the Ridge classifier, which is linear regression with regulization
def main():
    print(len(X))
    print(len(y))
    for name, classifier in classifiers.items():
        print(name)
        #classifier.fit(X, y)
        #performance = -cross_val_score(classifier, X, y, scoring="neg_mean_squared_error", cv=13).mean()
        performance = -cross_val_score(classifier, X, y, scoring="neg_mean_squared_error", cv=13).mean()
        print("\t", performance)
    performance = -cross_val_score(l, X, y, scoring="neg_mean_squared_error", cv=13).mean()
    print("sklearn lr: ", performance)
if __name__ == "__main__":
  main()
            