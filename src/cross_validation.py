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
    "greedy": GreedyLinearRegression(alpha=alpha, iterations=iterations),
    "linear regression 2.0": LinearRegression(alpha=alpha * 2, iterations=iterations),
    "greedy 2.0": GreedyLinearRegression(alpha=alpha *2, iterations=iterations),
    "linear regression .5" : LinearRegression(alpha = alpha / 2, iterations = iterations),
    "greedy .5" : GreedyLinearRegression(alpha= alpha / 2, iterations = iterations),
    "linear .25": LinearRegression(alpha = alpha / 4, iterations = iterations),
    "greedy .25": GreedyLinearRegression(alpha = alpha / 4, iterations = iterations),
    "linear .125": LinearRegression(alpha = alpha / 8, iterations = iterations),
    "greedy .125": GreedyLinearRegression(alpha = alpha / 8, iterations = iterations),
    
    "linear regression iter = 500": LinearRegression(alpha=alpha, iterations=500),
    "greedy iter 500": GreedyLinearRegression(alpha=alpha, iterations=500),
    "linear regression 2.0 iter 500": LinearRegression(alpha=alpha * 2, iterations=500),
    "greedy 2.0 iter 500": GreedyLinearRegression(alpha=alpha *2, iterations=500),
    "linear regression .5 iter 500" : LinearRegression(alpha = alpha / 2, iterations = 500),
    "greedy .5 iter 500" : GreedyLinearRegression(alpha= alpha / 2, iterations = 500),
    "linear .25 iter 500": LinearRegression(alpha = alpha / 4, iterations = 500),
    "greedy .25 iter 500": GreedyLinearRegression(alpha = alpha / 4, iterations = 500),
    "linear .125 iter 500": LinearRegression(alpha = alpha / 8, iterations = 500),
    "greedy .125 iter 500": GreedyLinearRegression(alpha = alpha / 8, iterations = 500),

    "linear regression iter = 1000": LinearRegression(alpha=alpha, iterations=1000),
    "greedy iter 1000": GreedyLinearRegression(alpha=alpha, iterations=1000),
    "linear regression 2.0 iter 1000": LinearRegression(alpha=alpha * 2, iterations=1000),
    "greedy 2.0 iter 1000": GreedyLinearRegression(alpha=alpha *2, iterations=1000),
    "linear regression .5 iter 1000" : LinearRegression(alpha = alpha / 2, iterations = 1000),
    "greedy .5 iter 1000" : GreedyLinearRegression(alpha= alpha / 2, iterations = 1000),
    "linear .25 iter 1000": LinearRegression(alpha = alpha / 4, iterations = 1000),
    "greedy .25 iter 1000": GreedyLinearRegression(alpha = alpha / 4, iterations = 1000),
    "linear .125 iter 1000": LinearRegression(alpha = alpha / 8, iterations = 1000),
    "greedy .125 iter 1000": GreedyLinearRegression(alpha = alpha / 8, iterations = 1000)

}

# plot the learning curve for simple linear regression and the Ridge classifier, which is linear regression with regulization
def main():
    print(len(X))
    print(len(y))
    for name, classifier in classifiers.items():
        print(name)
        #classifier.fit(X, y)
        #performance = -cross_val_score(classifier, X, y, scoring="neg_mean_squared_error", cv=13).mean()
        performance = -cross_val_score(classifier, X, y, scoring="neg_mean_squared_error", cv=13)
        print("\t", performance)
    performance = -cross_val_score(l, X, y, scoring="neg_mean_squared_error", cv=13).mean()
    print("sklearn lr: ", performance)
if __name__ == "__main__":
  main()
            