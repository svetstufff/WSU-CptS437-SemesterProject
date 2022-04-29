import numpy as np
from sklearn.base import BaseEstimator

# simple linear regression model
class GreedyLinearRegression(BaseEstimator):
    def __init__(self, alpha = 0.001, iterations = 100):
        self.alpha = alpha # learning rate
        self.iterations = iterations # number of epochs used to learn the coefficient vector
        self.name = "greedy"
    
    # returns the the the square of the magnitude of each feature vector
    # thetas are updated in order of magnitude of their corresponding feature vector 
    # these magnitudes are used to determine this sorted order
    @staticmethod
    def __sum_squared_features(X, n, k):
        sum_squares = np.zeros(k+1)
        sum_squares[-1] = 1
        for i in range(n):
            for j in range(k):
                sum_squares[j] += X[i][j]**2
        return sum_squares
        
    # learns a linear function that fits the relationship between the provided feature and class values
    def fit(self, X, y):
        # terms collected before each epoch - these are used to plot thetas vs. loss in 3D (see loss_paths.py)
        bias_path = np.zeros(self.iterations) # value of bias term
        theta_1_path = np.zeros(self.iterations) # value of first theta coefficient
        loss_path = np.zeros(self.iterations) # mean squared error before each epoch

        n = len(y) # size of dataset, number of datapoints
        k = len(X[0]) # dimensionality, number of features
        self.thetas = np.zeros(k+1) # learned linear function represented as a vector of coefficients (initially, all coefficients = 0)
        predicted = np.zeros(n) # predicted class values for each datapoint based on current linear function - as linear function gets updated, these get updated as well

        # get squares of magnitudes of feature vectors and sort thetas according to these magnitudes
        sum_squares = self.__sum_squared_features(X, n, k)

        # this is the order in which thetas are updated
        # note that each "theta" refers to the index of their associated feature
        # e.g., if a theta is responsible for multiplying the 2nd feature, theta = 1
        sorted_thetas = np.argsort(sum_squares)

        # for each epoch, predict the class values for each datapoint
        # use predicted and ground truth class values in gradient descent method to update coefficient vector
        epochs = 0
        while epochs < self.iterations:
            # update current theta values
            bias_path[epochs] = self.thetas[0]
            theta_1_path[epochs] = self.thetas[1]

            # apply current linear function to each datapoint to compute predicted class values
            for i in range(n):
                predicted[i] = self.thetas[-1]
                for j in range(k): # each feature has its own coefficient
                    predicted[i] += self.thetas[j] * X[i][j]
                loss_path[epochs] += (y[i] - predicted[i]) ** 2 # update mean squared errors

            # compute mean by dividing sum of squared errors by num datapoints
            loss_path[epochs] /= n    
            
            # for each epoch, update all theta parameters one by one in order of sorted_thetas
            theta_adjustments = np.zeros(k+1) # keeps track of adjustment made to each theta parameter - used to incorporate adjustment into gradient calculation of subsequent thetas
            
            # iterate through thetas in sorted order
            for sorted_index, theta in enumerate(sorted_thetas):
                gradient = 0 # gradient for current theta
                for i in range(n):
                    if (sorted_index > 0): # incorporate adjustment made to previous theta into current predicted class value
                        prev_theta = sorted_thetas[sorted_index - 1] # get last theta parameter adjusted
                        if prev_theta == k: # if this is the bias term, simply add adjustment
                            predicted[i] += theta_adjustments[prev_theta]
                        else: # if coefficient term, multiply adjustment by corresponding feature value
                            predicted[i] += X[i][prev_theta] * theta_adjustments[prev_theta]
                    diff = y[i] - predicted[i]
                    if theta == k: # calculation of gradient is different for bias term vs. coefficient term
                        gradient += diff
                    else:
                        gradient += diff * X[i][theta]
                self.thetas[theta] += self.alpha * gradient # adjust theta value using learning rate
                theta_adjustments[theta] = self.alpha * gradient                    
                                
            epochs += 1        
        
        return bias_path, theta_1_path, loss_path
        

    # returns an array of predicted class values for each feature vector in X
    # uses linear function (represented by coefficient vector self.thetas) learned in fit()
    def predict(self, X):
        n = len(X) # number of datapoints
        k = len(X[0]) # number of features
        predicted = [0] * n

        # iterate through each feature vector and apply learned linear function to calculate predicted class value
        for i in range(n):
            predicted[i] = self.thetas[-1] # bias term is the last theta parameter 
            for j in range(k): # include coefficient assigned to each feature
                predicted[i] += self.thetas[j] * X[i][j]

        return predicted
        
