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
