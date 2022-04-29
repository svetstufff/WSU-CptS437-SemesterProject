from data import X_diabetes, y_diabetes, X_cali, y_cali, X_linnerud, y_linnerud
from bootstrap import bootstrap
from cross_validation import cross_validation
from loss_paths import loss_paths
from runtime import runtime
from helper import clear_console

def main():
    clear_console()
    
    # invoke the function associated with each evaluation metric

    # bootstrapping diabetes
    bootstrap(X_diabetes, y_diabetes, "diabetes", hyperparameters = {
        "alpha": 0.004,
        "iterations": 100
    })

    # boostrapping california housing
    # note we only use the first 100 instances to limit runtime
    bootstrap(X_cali[:100], y_cali[:100], "california housing", hyperparameters = {
        "alpha": 0.009,
        "iterations": 100
    })

    # boostrapping linnerud 
    bootstrap(X_linnerud, y_linnerud, "linnerud", hyperparameters = {
        "alpha": 0.04,
        "iterations": 50
    })
    
    # cross validation
    cross_validation()

    # loss paths
    loss_paths(hyperparameters = {
         "alpha": 0.00393,
         "iterations": 100
    })

    # runtime
    runtime()

main()