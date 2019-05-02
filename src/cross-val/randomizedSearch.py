import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from jupyterthemes import jtplot
jtplot.style(theme="monokai")
from warnings import filterwarnings
filterwarnings("ignore")


# Cross Validation Function for Random Forest
def RandomForest(x_df, y_df, cv, n):
    model = RandomForestClassifier(random_state=100, verbose=True)
    params = {"n_estimators": [200, 400, 600, 800, 1000],
              "max_depth": [3, None],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=n)
    random_fit = random_search.fit(x_df, y_df)
    print(random_fit.cv_results_)
    print(f"Best Cross-Validated Accuracy: {round(random_fit.best_score_()*100,2)}%")
    print(f"Best Model Parameters: {best_params_}")
    return random_fit

# Cross Validation Function for Logistic Regression Non-SGD
def noSGDLogisticRegression(x_df, y_df, cv, n):
    model = LogisticRegression(random_state=100, verbose=True)
    params = {"penalty": ['l1', 'l2'],
              "fit_intercept": [False, True],
              "intercept_scaling": [1, 1.25, 1.5, 1.75, 2]}
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=n)
    random_fit = random_search.fit(x_df, y_df)
    cv_scores = cross_val_score(model, x_df, y_df, cv=cv)
    print(cv_scores)
    print(f"Mean accuracy: {round(cv_scores.mean()*100,2)}%")
    print(f"Standard Deviation of Accuracies: {round(cv_scores.std()*100,2)}%")
    return random_fit

# Cross Validation Function for Logistic Regression SGD

def SGDLogisticRegression(x_df, y_df, cv, n):
    model = SGDClassifier(loss='log', random_state=100, verbose=True)
    params = {"penalty": ['l1', 'l2', 'elasticnet'],
              "l1_ratio": [0.15, 0.25, 0.45, 0.85, 1],
              "shuffle": [True, False],
              "learning_rate": ['constant', 'optimal', 'invscaling', 'adaptive'],
              "eta0": [0.0, 0.15, 0.20, 0.25, 0.75, 0.85],
              "early_stopping": [True, False]}
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=n)
    random_fit = random_search.fit(x_df, y_df)
    cv_scores = cross_val_score(model, x_df, y_df, cv=cv)
    print(cv_scores)
    print(f"Mean accuracy: {round(cv_scores.mean()*100,2)}%")
    print(f"Standard Deviation of Accuracies: {round(cv_scores.std()*100,2)}%")
    return random_fit

# Cross Validation Function for Neural Network

def NeuralNetwork(x_df, y_df, cv, n):
    model = MLPClassifier(random_state=100, verbose=True)
    params = {"hidden_layer_sizes": [(100,), (150,), (200,), (250,), (300,)],
              "activation": ['logistic', 'tanh', 'relu'],
              "solver": ['lbfgs', 'sgd', 'adam'],
              "alpha": [1e-4, 1e-8, 1e-12, 1e-18],
              "learning_rate": ['constant', 'adaptive'],
              "shuffle": [True, False]
              }
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=n)
    random_fit = random_search.fit(x_df, y_df)
    cv_scores = cross_val_score(model, x_df, y_df, cv=cv)
    print(cv_scores)
    print(f"Mean accuracy: {round(cv_scores.mean()*100,2)}%")
    print(f"Standard Deviation of Accuracies: {round(cv_scores.std()*100,2)}%")
    return random_fit
