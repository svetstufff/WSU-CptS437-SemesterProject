# performs a t-test on two samples of cross-validation mean squared errors 
# returns the range of the resulting p-value, which denotes the statistical significance of the difference in sample means
import scipy.stats
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
def ttest(n, mean_squared_errors_1, mean_squared_errors_2):
    # find means of mean squared error values

    # convert mean squared error values to deviations from their respective means

    # use formula to compute t-value

    # convert t-value to range of p-values using table
    l_err1 = list(mean_squared_errors_1)
    l_err2 = list(mean_squared_errors_2)
    sum_errors_1 = 0.0
    sum_errors_2 = 0.0
    for i in l_err1:
        sum_errors_1 += i
    for i in l_err2:
        sum_errors_2 +=i
    
    mu_a = sum_errors_1 / n
    mu_b = sum_errors_2 / n

    
    i = 0
    sum_ahat_minus_bhat_sqrd = 0
    while i < n:
        sum_ahat_minus_bhat_sqrd += (mu_a - l_err1[i] - mu_b - l_err2[i]) ** 2
        i += 1
    
    t = (mu_a - mu_b) * ((n*(n-1) / sum_ahat_minus_bhat_sqrd) ** 1/2)

        

    p_value = scipy.stats.t.sf(abs(t), df = n-1)*2



    return p_value


def bt_conf(percent, X, y, classifier, num_iter):
    
    n_iterations = num_iter
    n = len(y)
    stats = list()
    for i in range(n_iterations):
        samples = np.random.randint(n, size=n)
        X_train = X[samples]
        y_train = y[samples]
        test = np.random.randint(n, size=n//4)
        X_test = X[test] 
        y_test = y[test]
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        score = accuracy_score(y_test, predictions)
        stats.append(score)
    plt.hist(stats)
    plt.show()
    p = ((1.0-percent)/2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (percent+((1.0-percent)/2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print(percent, lower, upper)