import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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
from warnings import filterwarnings
filterwarnings("ignore")
# Function Approach for Our Data

neurFeat = pd.read_csv("data/Merged/merged_labeled_DevAttentionX.csv")
neurFeat.drop(list(neurFeat.columns)[0], axis=1, inplace=True)
classValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv")
classValues.drop(list(classValues.columns)[0], axis=1, inplace=True)

# Scale Features
zFeat = scale(neurFeat)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(zFeat, classValues, test_size=0.2, random_state=100)

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

pred = RandomForest(Xtrain, Ytrain, 5, 5)
plot_roc(Ytest, pred.predict_proba(Xtest), title="RF ROC")
