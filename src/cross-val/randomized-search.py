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
def RandomForest(x_df, y_df, cv, n_iter):
    features = x_df
    class_labels = y_df
    zFeat = scale(features)
    X_train, X_test, Y_train, Y_test = train_test_split(zFeat, class_labels, test_size=0.20, random_state=100)
    model = RandomForestClassifier(random_state=100)
    cv_results = cross_val_score(model, X_train, Y_train, cv=cv)
    params = {"n_estimators": [200, 400, 600, 800, 1000],
              "max_depth": [3, None],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=cv)
    random_fit = random_search.fit(zFeat, class_labels)
    # ROC Curve
    plot_roc(Ytest, random_fit.predict_proba(Xtest), title="CV Random Forest ROC")
    # Confusion Matrix
    plot_confusion_matrix(Ytest, random_fit.predict(Xtest), title="CV Random Forest CF")
    return f"Model accuracy: {round(accuracy_score(Ytest, random_fit.predict(Xtest))*100, 2)}"

pred = RandomForest(neurFeat, classValues, 5, 5)
