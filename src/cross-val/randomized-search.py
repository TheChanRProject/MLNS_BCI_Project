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

# Report Function

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

# Cross Validation Function for Random Forest
def RandomForest(x_df, y_df, cv, n_estimators, random_state, n_iter):
    NeurFeat = x_df
    classValues = y_df
    zFeat = scale(NeurFeat)
    X_train, X_test, Y_train, Y_test = train_test_split(zFeat, classValues, test_size=0.20, random_state=100)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    cv_results = cross_val_score(model, X_train, Y_train, cv=cv)
    param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    random_search = RandomizedSearchCV(model, n_iter=n_iter, cv=cv)
    random_fit = random_search.fit(zFeat, classValues)
    # ROC Curve
    plot_roc(Ytest, random_fit.predict_proba(X_train))
    plot_confusion_matrix(Ytest, random_fit.predict(X_train))
    return report(random_fit.cv_results_)

RandomForest(neurFeat, classValues, 5, 400, 100, 5)
