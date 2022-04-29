# loss_paths.py
# as linear function approaches local minima, the gradients get smaller and smaller - greedy adjusts one term first, which lowers the gradient of the other, allowing for a more gradual approach
# in general, this is why lower learning rates = worse performance for greedy; it will simply adjust slower to towards the local minima, without jumping over it
hyperparameters = {
    "alpha": 0.00393,
    "iterations": 100
}

hyperparameters = {
    "alpha": 0.00395,
    "iterations": 100
}

# cross validation
ranges = {
    "alpha": (0.005, 0.02),
    "iterations": (50, 100)
}

hyperparameter_values = {
    "alpha": 0.02,
    "iterations": 100
}

# bootstrap
hyperparameters = {
    "alpha": 0.004,
    "iterations": 100
}
num_resamples = 1000