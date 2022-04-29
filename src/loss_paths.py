from data import X_one_feature as X, y_one_feature as y
from LinearRegression import LinearRegression
from GreedyLinearRegression import GreedyLinearRegression
import matplotlib.pyplot as plt
from helper import save_graph

def loss_paths(hyperparameters):
    print('LOSS PATHS')

    # defines the hyperparameters used for each classifier

    # initialize both classifiers using hyperparameters above
    linear = LinearRegression(**hyperparameters)
    greedy = GreedyLinearRegression(**hyperparameters)

    # fit to dataset and store loss after each epoch
    linear_bias_path, linear_theta_1_path, linear_loss_path = linear.fit(X, y)
    greedy_bias_path, greedy_theta_1_path, greedy_loss_path = greedy.fit(X, y)

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

    # plot loss paths against bias term and theta_1

    # plot 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(linear_bias_path, linear_theta_1_path, linear_loss_path, label='linear', c='red')
    plt.legend()
    ax.plot(linear_bias_path, linear_theta_1_path, linear_loss_path, c='red')
    ax.scatter(greedy_bias_path, greedy_theta_1_path, greedy_loss_path, label='greedy', c='green')
    plt.legend()
    ax.plot(greedy_bias_path, greedy_theta_1_path, greedy_loss_path, c='green')
    plt.show()
    save_graph(fig, f'loss_paths_3D')

    
loss_paths(hyperparameters={ "alpha": 0.00393, "iterations": 100 })
