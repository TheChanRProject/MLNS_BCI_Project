import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
filterwarnings("ignore")

# Cross Validation Function for Random Forest
def RandomForest(x_df, y_df, cv, n):
    model = RandomForestClassifier(random_state=100, verbose=True)
    params = {"n_estimators": [400, 1000],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    grid_search = GridSearchCV(model, param_distributions=params, cv=cv, n_iter=n)
    grid_fit = grid_search.fit(x_df, y_df)
    cv_scores = cross_val_score(model, x_df, y_df, cv=cv)
    print(cv_scores)
    print(f"Mean accuracy: {round(cv_scores.mean()*100,2)}%")
    print(f"Standard Deviation of Accuracies: {round(cv_scores.std()*100,2)}%")
    return random_fit
