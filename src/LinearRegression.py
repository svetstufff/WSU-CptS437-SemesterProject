import numpy as np
from sklearn.base import BaseEstimator

# simple linear regression model
class LinearRegression(BaseEstimator):
    def __init__(self, alpha = 0.001, iterations = 100):
        self.alpha = alpha # learning rate
        self.iterations = iterations # number of epochs used to learn the coefficient vector\
        self.name = "linear"

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

        # for each epoch, predict the class values for each datapoint
        # use predicted and ground truth class values in gradient descent method to update coefficient vector
        epochs = 0
        while epochs < self.iterations:
            # update current theta values
            bias_path[epochs] = self.thetas[0]
            theta_1_path[epochs] = self.thetas[1]

            # apply current linear function to each datapoint to compute predicted class values
            for i in range(n):
                predicted[i] = self.thetas[0]
                for j in range(k): # each feature has its own coefficient
                    predicted[i] += self.thetas[j+1] * X[i][j]
                    
            # apply gradient descent by computing partial derivatives
            total_diff = np.zeros(k+1) # array of partial derivative for each feature
            for i in range(n):
                diff = y[i] - predicted[i]      # y - h(x)
                total_diff[0] += diff         # Update Theta_0 by diff
                loss_path[epochs] += diff ** 2 # update mean squared error for current epoch
                for j in range(k):
                    total_diff[j+1] += diff * X[i][j]   # Update Theta_1 by diff * x
            
            # update coefficients using learning rate and partial derivatives computed above 
            for j in range(k+1):
                self.thetas[j] += self.alpha * total_diff[j]

            # compute mean by dividing sum of squared errors by num datapoints
            loss_path[epochs] /= n    

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
            predicted[i] = self.thetas[0]
            for j in range(k): # include coefficient assigned to each feature
                predicted[i] += self.thetas[j+1] * X[i][j]

        return predicted
        
