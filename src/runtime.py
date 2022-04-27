from data import X_diabetes as X, y_diabetes as y
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression
import matplotlib.pyplot as plt
from helper import show_progress, save_graph
import timeit

def runtime():
    tested_iterations_values = [i for i in range(1, 100)]
    runtimes = {
        "linear": [],
        "greedy": []
    }
    i = 0
    for iterations in tested_iterations_values:
        linear = LinearRegression(iterations=iterations)
        greedy = GreedyLinearRegression(iterations=iterations)
        classifiers = [linear, greedy]

        for classifier in classifiers:
            start = timeit.default_timer()
            classifier.fit(X, y)
            stop = timeit.default_timer()
            runtimes[classifier.name].append(stop - start)

        show_progress(i, len(tested_iterations_values))
        i += 1

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