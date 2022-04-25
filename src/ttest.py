# performs a t-test on two samples of cross-validation mean squared errors 
# returns the range of the resulting p-value, which denotes the statistical significance of the difference in sample means
import scipy.stats
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
    
    t = (mu_a - mu_b) ((n(n-1) / sum_ahat_minus_bhat_sqrd) ** 1/2)

        

    p_value = scipy.stats.t.sf(abs(t), df = n-1)*2
    


    return p_value_min, p_value_max