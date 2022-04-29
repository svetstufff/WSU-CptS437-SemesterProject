from data import X_diabetes, y_diabetes
from bootstrap import bootstrap
from cross_validation import cross_validation
from loss_paths import loss_paths
from runtime import runtime
from helper import clear_console

def main():
    clear_console()
    
    # invoke the function associated with each evaluation metric
    # bootstrapping
    bootstrap(X_diabetes, y_diabetes, "diabetes", hyperparameters = {
        "alpha": 0.004,
        "iterations": 100
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