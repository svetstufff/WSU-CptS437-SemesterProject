import numpy as np
from sklearn.base import BaseEstimator

# simple linear regression model
class GreedyLinearRegression(BaseEstimator):
    def __init__(self, alpha = 0.001, iterations = 100):
        self.alpha = alpha # learning rate
        self.iterations = iterations # number of epochs used to learn the coefficient vector
    
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
        n = len(y) # size of dataset, number of datapoints
        k = len(X[0]) # dimensionality, number of features
        self.thetas = np.zeros(k+1) # learned linear function represented as a vector of coefficients (initially, all coefficients = 0)
        predicted = np.zeros(n) # predicted class values for each datapoint based on current linear function - as linear function gets updated, these get updated as well

        sum_squares = self.__sum_squared_features(X, n, k)
        sorted_thetas = np.argsort(sum_squares)

        # for each epoch, predict the class values for each datapoint
        # use predicted and ground truth class values in gradient descent method to update coefficient vector
        epochs = 0
        while epochs < self.iterations:
            # apply current linear function to each datapoint to compute predicted class values
            for i in range(n):
                predicted[i] = self.thetas[-1]
                for j in range(k): # each feature has its own coefficient
                    predicted[i] += self.thetas[j] * X[i][j]
            
            theta_adjustments = np.zeros(k+1)
            for sorted_index, theta in enumerate(sorted_thetas):
                gradient = 0
                for i in range(n):
                    if (sorted_index > 0):
                        prev_theta = sorted_thetas[sorted_index - 1]
                        if prev_theta == k:
                            predicted[i] += theta_adjustments[prev_theta]
                        else:
                            predicted[i] += X[i][prev_theta] * theta_adjustments[prev_theta]
                    diff = y[i] - predicted[i]
                    if theta == k:
                        gradient += diff
                    else:
                        gradient += diff * X[i][theta]
                self.thetas[theta] += self.alpha * gradient
                theta_adjustments[theta] = self.alpha * gradient                    
                                
            epochs += 1        
        

    # returns an array of predicted class values for each feature vector in X
    # uses linear function (represented by coefficient vector self.thetas) learned in fit()
    def predict(self, X):
        n = len(X) # number of datapoints
        k = len(X[0]) # number of features
        predicted = [0] * n

        # iterate through each feature vector and apply learned linear function to calculate predicted class value
        for i in range(n):
            predicted[i] = self.thetas[-1]
            for j in range(k): # include coefficient assigned to each feature
                predicted[i] += self.thetas[j] * X[i][j]

        return predicted
        
