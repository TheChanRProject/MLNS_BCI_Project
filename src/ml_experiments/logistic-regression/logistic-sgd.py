# This is the code for implementing a logistic regression with stochastic gradient descent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from warnings import filterwarnings
filterwarnings("ignore")

# Reading Neural Features and Outputs
neuralFeatures = pd.read_csv("data/Merged/merged_labeled_DevAttentionX.csv").loc[0:119999]
neuralFeatures.drop(list(neuralFeatures.columns)[0], axis=1, inplace=True)
classValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv").loc[0:119999]
classValues.drop(list(classValues.columns)[0], axis=1, inplace=True)

# Scale features
zFeatures = scale(neuralFeatures, axis=0)

# Splitting into Training and Testing Sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(zFeatures, classValues, test_size=0.2, random_state=100)

# Logistic Regression Classifier with Stochastic Gradient Descent
log_model = SGDClassifier(loss='log', penalty='none')
log_fit = log_model.fit(Xtrain, Ytrain)
log_pred_test = log_fit.predict(Xtest)
log_pred_test_prob = log_fit.predict_proba(Xtest)

# Plot ROC curve and confusion matrix
from jupyterthemes import jtplot
jtplot.style(theme="monokai")

plot_roc(Ytest, log_pred_test_prob, title="Logistic Regression SGD ROC")
plot_confusion_matrix(Ytest, log_pred_test, title="Logistic Regression SGD Confusion Matrix")
print(f"Accuracy Score: {round(accuracy_score(Ytest, log_pred_test)*100,2)}%")

# Logistic Regression with SGD and L2 Penalty
log_l2_model = SGDClassifier(loss='log', penalty='l2')
log_l2_fit = log_l2_model.fit(Xtrain, Ytrain)
log_l2_test_pred = log_l2_fit.predict(Xtest)
log_l2_test_pred_prob = log_l2_fit.predict_proba(Xtest)

# Plot ROC and Confusion Matrix
plot_roc(Ytest, log_l2_test_pred_prob, title="Logistic Regression with SGD and L2 ROC")
plot_confusion_matrix(Ytest, log_l2_test_pred, title="Logistic Regression with SGD and L2 Confusion Matrix")
print(f"Accuracy Score: {round(accuracy_score(Ytest, log_l2_test_pred)*100,2)}%")
