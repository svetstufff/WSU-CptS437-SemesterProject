from data import X_diabetes as X, y_diabetes as y
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression
import matplotlib.pyplot as plt
from helper import show_progress, save_graph
import timeit

# reports the runtime of the two classifiers for different values of the iterations hyperparameter
def runtime():
    print('RUNTIME')

    # test iterations 1 - 100
    tested_iterations_values = [i for i in range(1, 100)]

    # each classifier is assigned an array of 100 runtime values, one for each iterations hyperparameter value
    runtimes = {
        "linear": [],
        "greedy": []
    }
    i = 0 # used to track progress

    # iterate through all iterations values tested, storing runtime for each classifier
    for iterations in tested_iterations_values:
        # initialize classifiers with provided iterations hyperparameter
        # note alpha will be set to default (0.001)
        linear = LinearRegression(iterations=iterations)
        greedy = GreedyLinearRegression(iterations=iterations)
        classifiers = [linear, greedy]

        # calculate and record training runtime for each classifier
        for classifier in classifiers:
            start = timeit.default_timer()
            classifier.fit(X, y)
            stop = timeit.default_timer()
            runtimes[classifier.name].append(stop - start)

        # show progress
        show_progress(i, len(tested_iterations_values))
        i += 1

    # plot runtimes and save figure
    fig, _ = plt.subplots()   
    plt.suptitle(f'Linear vs. greedy runtimes')
    plt.xlabel('iterations')
    plt.ylabel('runtime (s)')
    plt.plot(tested_iterations_values, runtimes["linear"], color="red", linewidth=2)
    plt.plot(tested_iterations_values, runtimes["greedy"], color="green", linewidth=2)
    plt.legend(["linear", "greedy"])
    plt.show()
    save_graph(fig, f'runtime')
        
runtime()