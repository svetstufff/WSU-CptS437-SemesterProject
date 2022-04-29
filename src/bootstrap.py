from helper import show_progress, save_graph
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from data import X_diabetes as X, y_diabetes as y
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression

# constructs a confidence interval for the mean squared error of the provided regression classifier
# employs the bootstrapping method to compute error values for many resamples of the provided dataset
# returns the lower and upper bound of confidence interval + "stats", which are the mean squared error values for each resampled dataset
def bootstrap(percent, X, y, classifier, num_resamples):
    print("\t|", classifier.name)
    n = len(y) # size of dataset
    stats = [] # stores the mean squared error values for each resampling

    # perform num_resamples resamples
    # each resample consists of taking a random sample of training and test data
    # from the dataset with replacement, training the classifier on the training sample,
    # and finally computing mean squared error on the test sample
    for i in range(num_resamples):
        # take a random sample of training instances with replacement
        resamples = np.random.randint(n, size=n)
        X_train = X[resamples]
        y_train = y[resamples]

        # take a random sample of test instances with replacement
        test = np.random.randint(n, size=n // 4)
        X_test = X[test] 
        y_test = y[test]

        # train classifier on training sample
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        # compute and store mean squared error of test sample
        score = mean_squared_error(y_test, predictions)
        stats.append(score)

        # show progrress
        show_progress(i, num_resamples)

    # compute lower and upper bound of confidence interval
    lower_percentile = (100 - percent) / 2
    lower = np.percentile(stats, lower_percentile)
    upper_percentile = 100 - lower_percentile
    upper = np.percentile(stats, upper_percentile)

    # return confidence interval + mean squared error values
    return [lower, upper], stats

def plot_bootstrap(X, y, hyperparameters):
    print("BOOTSTRAP")
    percent = 95
    num_resamples = 1000

    # initialize both classifiers using hyperparameters above
    linear = LinearRegression(**hyperparameters)
    greedy = GreedyLinearRegression(**hyperparameters)
    classifiers = [linear, greedy]

    # plot histogram of bootstrapped mean squared errors w/ superimposed confidence intervals
    classifier_colors = {
        "linear": "red",
        "greedy": "green"
    }
    fig, _ = plt.subplots()
    plt.suptitle(f'Linear vs. greedy: bootstrapped mean squared errors')
    plt.title(f'alpha={hyperparameters["alpha"]}, iterations={hyperparameters["iterations"]}')
    plt.xlabel('mean squared error')
    plt.ylabel('frequency')
    alph = 1
    for classifier in classifiers:
        interval, stats = bootstrap(percent=percent, X=X, y=y, classifier=classifier, num_resamples=num_resamples)

        color = classifier_colors[classifier.name]

        # plot histogram of bootstrapped errors
        (counts, _, _) = plt.hist(stats, color=color, label=classifier.name, alpha = alph)
        plt.legend()
        interval_height = [np.max(counts) * 1.25] * 2 # height of each interval will be set to 10% larger than largest count across all bins

        # plot confidence interval
        plt.plot(interval, interval_height, color=color) # plot line
        plt.scatter(interval, interval_height, color=color) # plot lower and upper bounds
        plt.scatter(np.mean(stats), interval_height[:1], color=classifier_colors[classifier.name]) # plot center
        alph -=.2
    # display and save graph

    plt.show()
    save_graph(fig, f'confidence_intervals')

plot_bootstrap()