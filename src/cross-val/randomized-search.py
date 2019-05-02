import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
from sklearn.model_selection import cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from jupyterthemes import jtplot
jtplot.style(theme="monokai")
from sklearn.tree import export_graphviz
import pydot

# Function Approach for Our Data

neurFeat = pd.read_csv("data/Merged/merged_labeled_DevAttentionX.csv")
neurFeat.drop(list(neurFeat.columns)[0], axis=1, inplace=True)
classValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv")
classValues.drop(list(classValues.columns)[0], axis=1, inplace=True)

# Cross Validation Function for Random Forest
def RandomForest(x_df, y_df, cv, n_estimators, random_state):
    NeurFeat = x_df
    classValues = y_df
    zFeat = scale(NeurFeat)
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, bootstrap=bootstrap, random_state=random_state)
    cv_results = cross_val_score(model, zFeat, classValues, cv=cv)
    param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    random_search = RandomizedSearchCV(model, )
