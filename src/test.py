from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression


# load diabetes dataset and define classifiers
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
    for name, classifier in classifiers.items():
        print(name)
        classifier.fit(X, y)
        performance = -cross_val_score(classifier, X, y, scoring="neg_mean_squared_error", cv=10).mean()
        print("\t", performance)
  
if __name__ == "__main__":
  main()
            