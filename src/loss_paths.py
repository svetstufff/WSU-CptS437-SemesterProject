from data import X, y
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression
import matplotlib.pyplot as plt
from helper import save_graph

def loss_paths():
    # defines the hyperparameters used for each classifier
    hyperparameters = {
        "alpha": 0.03,
        "iterations": 200
    }

    # initialize both classifiers using hyperparameters above
    linear = LinearRegression(**hyperparameters)
    greedy = GreedyLinearRegression(**hyperparameters)

    # fit to dataset and store loss after each epoch
    linear_loss_path = linear.fit(X,y)
    greedy_loss_path = greedy.fit(X, y)

    # plot loss paths
    fig, _ = plt.subplots()   
    plt.suptitle(f'Linear vs. greedy loss paths')
    plt.title('value of loss function after each epoch')
    plt.xlabel('epoch')
    plt.ylabel('mean squared error')
    plt.plot(range(hyperparameters["iterations"]), linear_loss_path, color="red", linewidth=2)
    plt.plot(range(hyperparameters["iterations"]), greedy_loss_path, color="green", linewidth=2)
    plt.legend(["linear", "greedy"])
    plt.show()
    save_graph(fig, f'loss_paths')
    

loss_paths()