from data import X_diabetes as X, y_diabetes as y
from sklearn.model_selection import cross_val_score
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression
import matplotlib.pyplot as plt
from helper import show_progress, save_graph
import scipy.stats
import numpy as np
import math

# performs a t-test on two samples of cross-validation mean squared errors 
# returns the range of the resulting p-value, which denotes the statistical significance of the difference in sample means
def ttest(n, errors_1, errors_2):
    # find means of mean squared error values
    mean_1 = np.mean(errors_1)
    mean_2 = np.mean(errors_2)

    # convert datapoints to deviations from respective means
    deviations_1 = [error - mean_1 for error in errors_1]
    deviations_2 = [error - mean_2 for error in errors_2]

    # compute denominator of t-value expression
    denominator = 0
    for deviation_1, deviation_2 in zip(deviations_1, deviations_2):
        denominator += (deviation_1 - deviation_2) ** 2 # sum of squared differences in deviation
        
    # use formula to compute t-value
    t = (mean_1 - mean_2) * math.sqrt((n * (n - 1)) / denominator)
    
    # convert t-value to p-value
    p = scipy.stats.t.sf(abs(t), df = n-1)*2

    return p

# takes in a range and a number of values tested and returns an array of tested values 
# int option is used to control rounding
def get_vals_tested(val_range, num_tested, int=False):
    vals_tested = [0] * num_tested
    increment = (val_range[1] - val_range[0]) / (num_tested - 1)
    val_tested = val_range[0]
    for i in range(num_tested):
        vals_tested[i] = val_tested if not int else math.floor(val_tested)
        val_tested += increment
    return vals_tested

def cross_validation():
    print('CROSS VALIDATION')

    # use only subset of data to control runtime
    global X
    global y
    X, y = X[:100], y[:100]

    # number of folds performed in all cross-validation tests
    n_folds = 5

    # names of hyperparameters tested
    hyperparameters_tested = ["alpha", "iterations"]

    # number of values tested for each hyperparameter
    num_hyperparameter_values_tested = 10

    # define ranges of hyperparameter values tested
    ranges = {
        "alpha": (0.005, 0.02),
        "iterations": (50, 100)
    }
    
    # create an array of tested values for each hyperparameter using the ranges above and the number of values tested
    vals_tested = {}
    for hyperparameter, range in ranges.items():
        vals_tested[hyperparameter] = get_vals_tested(range, num_hyperparameter_values_tested, int=(hyperparameter == "iterations")) # tested iteration values should be integers 

    # each classifier will have a dict of average mean squared errors 
    # the keys will be the hyperparameter, the values will be a dict - its keys will be the hyperparameter values tested, the values the mean cross validation score
    cross_val_means = {
        "linear": {},
        "greedy": {}
    }

    # dictionary of % differences in cross val scores across all hyperparameter values
    # keys are hyperparameters, values are dicts - each dict has hyperparameter values as keys, % differences as values
    cross_val_mean_differences = {}

    for hyperparameter in hyperparameters_tested:
        cross_val_mean_differences[hyperparameter] = {}

    # dictionary of p-values for cross val scores across all hyperparameter values
    # keys are hyperparameters, values are dicts - each dict has hyperparameter values as keys, p-value as values
    cross_val_p_values = {}

    for hyperparameter in hyperparameters_tested:
        cross_val_p_values[hyperparameter] = {}

    # dictionary of p-values in cross val scores across all 

    # populate structure of cross_val_means dict
    # for each hyperparameter, define a dict for each classifier - these will hold the cross val scores across all values tested
    for hyperparameter in hyperparameters_tested:
        cross_val_means["linear"][hyperparameter] = {}
        cross_val_means["greedy"][hyperparameter] = {}
    
    # defines hyperparameter values currently being tested - these are initially set to the default values
    hyperparameter_values = {
        "alpha": 0.02,
        "iterations": 100
    }

    # main loop - iterate through each hyperparameter tested and construct dict of cross val means for each value tested
    for hyperparameter in hyperparameters_tested:
        print("\t|", hyperparameter)
        default_val = hyperparameter_values[hyperparameter] # store default val - hyperparameter will be reset to this val once testing is complete
        i = 0 # used to track progress
        for val in vals_tested[hyperparameter]:
            # update hyperparameter val with new value being tested
            hyperparameter_values[hyperparameter] = val

            # initialize two classifiers using current hyperparameter values
            linear = LinearRegression(**hyperparameter_values)
            greedy = GreedyLinearRegression(**hyperparameter_values)

            # perform cross validation on both classifiers
            linear_cross_val_scores = -cross_val_score(linear, X, y, scoring="neg_mean_squared_error", cv=n_folds)
            greedy_cross_val_scores = -cross_val_score(greedy, X, y, scoring="neg_mean_squared_error", cv=n_folds)

            # store average mean squared error for both classifiers
            linear_performance = linear_cross_val_scores.mean()
            greedy_performance =  greedy_cross_val_scores.mean()

            # update cross val means
            cross_val_means["linear"][hyperparameter][val] = linear_performance
            cross_val_means["greedy"][hyperparameter][val] = greedy_performance

            # update % difference between cross val means
            cross_val_mean_differences[hyperparameter][val] = (linear_performance - greedy_performance) / linear_performance

            # update p-value
            cross_val_p_values[hyperparameter][val] = ttest(n_folds, linear_cross_val_scores, greedy_cross_val_scores)

            # show progress
            show_progress(i, num_hyperparameter_values_tested)
            i += 1

        # reset hyperparameter to default value for next tests
        hyperparameter_values[hyperparameter] = default_val


    # plot results

    # for each hyperparameter, plot % differences in errors between two classifiers across all values tested
    for hyperparameter in hyperparameters_tested:
        # plot cross val scrores across all hyperparameter values for linear and greedy
        fig, _ = plt.subplots()   
        plt.suptitle(f'Linear vs. greedy mean squared errors: {hyperparameter}')
        plt.title(f'average {n_folds}-fold cross-validation scores')
        plt.xlabel(hyperparameter)
        plt.ylabel('average mean squared error')
        plt.plot(vals_tested[hyperparameter], cross_val_means["linear"][hyperparameter].values(), color="red", linewidth=2)
        plt.plot(vals_tested[hyperparameter], cross_val_means["greedy"][hyperparameter].values(), color="green", linewidth=2)
        plt.legend(["linear", "greedy"])
        plt.show()
        save_graph(fig, f'{hyperparameter}')
    

        # plot percent differences across all values tested
        fig, _ = plt.subplots()
        plt.suptitle(f'Linear vs. greedy % differences: {hyperparameter}')
        plt.title(f'% difference between average {n_folds}-fold cross-validation scores')
        plt.xlabel(hyperparameter)
        plt.ylabel('% difference')
        plt.plot(vals_tested[hyperparameter], cross_val_mean_differences[hyperparameter].values(), color="blue", linewidth=2)
        plt.legend(["% difference"])
        plt.show()
        save_graph(fig, f'{hyperparameter}_diff')

        # plot p-values across all values tested
        fig, _ = plt.subplots()   
        plt.suptitle(f'Linear vs. greedy t-test: {hyperparameter}')
        plt.title(f'statistical significance of difference in average {n_folds}-fold cross-validation scores')
        plt.xlabel(hyperparameter)
        plt.ylabel('p-value')
        plt.plot(vals_tested[hyperparameter], cross_val_p_values[hyperparameter].values(), color="purple", linewidth=2)
        plt.legend(["p-value"])
        plt.show()
        save_graph(fig, f'{hyperparameter}_p')
        

cross_validation()
            