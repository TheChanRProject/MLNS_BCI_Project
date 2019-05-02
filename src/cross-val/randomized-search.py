import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
def RandomForest(x_df, y_df, cv, n_estimators, criterion, bootstrap, random_state):
    NeurFeat = x_df
    classValues = y_df
    zFeat = scale(NeurFeat)
    X_train, X_test, Y_train, Y_test = train_test_split(zFeat, classValues, test_size=0.0, random_state=100)
    model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, bootstrap=bootstrap, random_state=random_state)
    model_fit = model.fit()
