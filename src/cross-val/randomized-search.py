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
    model = RandomForestClassifier(random_state=100)
    params = {"n_estimators": [200, 400, 600, 800, 1000],
              "max_depth": [3, None],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=n)
    random_fit = random_search.fit(x_df, y_df)
    return random_fit
