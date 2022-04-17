import numpy as np
from sklearn.base import BaseEstimator

# simple linear regression model
class LinearRegression(BaseEstimator):
    def __init__(self, alpha = 0.001, iterations = 100):
        self.alpha = alpha # learning rate
        self.iterations = iterations # number of epochs used to learn the coefficient vector

    # learns a linear function that fits the relationship between the provided feature and class values
    def fit(self, X, y):
        n = len(y) # size of dataset, number of datapoints
        k = len(X[0]) # dimensionality, number of features
        self.thetas = np.zeros(k+1) # learned linear function represented as a vector of coefficients (initially, all coefficients = 0)
        predicted = np.zeros(n) # predicted class values for each datapoint based on current linear function - as linear function gets updated, these get updated as well

        # for each epoch, predict the class values for each datapoint
        # use predicted and ground truth class values in gradient descent method to update coefficient vector
        epochs = 0
        while epochs < self.iterations:
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
                for j in range(k):
                    total_diff[j+1] += diff * X[i][j]   # Update Theta_1 by diff * x
            


            # update coefficients using learning rate and partial derivatives computed above 
            for j in range(k+1):
                self.thetas[j] += self.alpha * total_diff[j]


            epochs += 1        

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
        
