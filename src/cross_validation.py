from statistics import linear_regression
from data import X, y
from sklearn.model_selection import cross_val_score
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression
import matplotlib.pyplot as plt

def clear_last_line():
    print ("\033[A                             \033[A")

def get_vals_tested(val_range, num_tested):
    return [val_range[0] + (i * (val_range[1] - val_range[0]) / (num_tested - 1)) for i in range(num_tested)]

def cross_validation():
    # number of folds performed in all cross-validation tests
    n_folds = 3

    # names of hyperparameters tested
    hyperparameters_tested = ["alpha", "iterations"]

    # number of values tested for each hyperparameter
    num_hyperparameter_values_tested = 20

    # define ranges of hyperparameter values tested
    ranges = {
        "alpha": (0.001, 0.01),
        "iterations": (25, 200)
    }
    
    # create an array of tested values for each hyperparameter using the ranges above and the number of values tested
    vals_tested = {}
    for hyperparameter, range in ranges.items():
        vals_tested[hyperparameter] = get_vals_tested(range, num_hyperparameter_values_tested)

    # each classifier will have a dict of average mean squared errors 
    # the keys will be the hyperparameter, the values will be a dict - its keys will be the hyperparameter values tested, the values the mean cross validation score
    cross_val_means = {
        "linear": {},
        "greedy": {}
    }

    cross_val_mean_differences = {
    }

    for hyperparameter in hyperparameters_tested:
        cross_val_mean_differences[hyperparameter] = {}

    

    # populate structure of cross_val_means dict
    # for each hyperparameter, define a dict for each classifier - these will hold the cross val scores across all values tested
    for hyperparameter in hyperparameters_tested:
        cross_val_means["linear"][hyperparameter] = {}
        cross_val_means["greedy"][hyperparameter] = {}
    
    # defines hyperparameter values currently being tested - these are initially set to the default values
    hyperparameter_values = {
        "alpha": 0.001,
        "iterations": 25
    }

    # main loop - iterate through each hyperparameter tested and construct dict of cross val means for each value tested
    for hyperparameter in hyperparameters_tested:
        print("|", hyperparameter)
        default_val = hyperparameter_values[hyperparameter] # store default val - hyperparameter will be reset to this val once testing is complete
        i = 0 # used to track progress
        for val in vals_tested[hyperparameter]:
            # update hyperparameter val with new value being tested
            hyperparameter_values[hyperparameter] = val

            # initialize two classifiers using current hyperparameter values
            linear = LinearRegression(**hyperparameter_values)
            greedy = GreedyLinearRegression(**hyperparameter_values)

            # perform cross validation and store mean error across all subsets
            linear_performance = -cross_val_score(linear, X, y, scoring="neg_mean_squared_error", cv=n_folds).mean()
            greedy_performance = -cross_val_score(greedy, X, y, scoring="neg_mean_squared_error", cv=n_folds).mean()

            # update cross val means
            cross_val_means["linear"][hyperparameter][val] = linear_performance
            cross_val_means["greedy"][hyperparameter][val] = greedy_performance

            # update difference between cross val means
            cross_val_mean_differences[hyperparameter][val] = 100 * (greedy_performance - linear_performance) / linear_performance


            # show progress
            i += 1
            if (i > 1):
                clear_last_line()
            print("\t", 100*(i / num_hyperparameter_values_tested), "%", sep="")


        # reset hyperparameter to default value for next tests
        hyperparameter_values[hyperparameter] = default_val


    # plot results

    # for each hyperparameter, plot % differences in errors between two classifiers across all values tested
    for hyperparameter in hyperparameters_tested:
        # plot cross val scrores across all hyperparameter values for linear and greedy
        plt.suptitle(f'Linear vs. greedy: {hyperparameter}')
        plt.title(f'average {n_folds}-fold cross-validation scores')
        plt.xlabel(hyperparameter)
        plt.ylabel('average mean squared error')
        plt.plot(vals_tested[hyperparameter], cross_val_means["linear"][hyperparameter].values(), color="red", linewidth=2)
        plt.plot(vals_tested[hyperparameter], cross_val_means["greedy"][hyperparameter].values(), color="green", linewidth=2)
        plt.legend(["linear", "greedy"])
        plt.show()      
        
        # plot percent differences across all values tested
        plt.suptitle(f'Linear vs. greedy: {hyperparameter}')
        plt.title(f'% difference between average {n_folds}-fold cross-validation scores')
        plt.xlabel(hyperparameter)
        plt.ylabel('% difference')
        plt.plot(vals_tested[hyperparameter], cross_val_mean_differences[hyperparameter].values(), color="blue", linewidth=2)
        plt.legend(["% difference"])
        plt.show()
        

cross_validation()
            