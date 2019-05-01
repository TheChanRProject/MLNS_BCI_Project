import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from warnings import filterwarnings
filterwarnings("ignore")
#from jupyterthemes import jtplot
#jtplot.style(theme='monokai')

# Preprocessing and Mapping Feature Labels

NeurFeats = pd.read_csv('/Users/teresaibarra/Downloads/merged_labeled_DevAttentionX.csv')
NeurFeats.drop(list(NeurFeats.columns)[0], axis=1, inplace=True)
NeurFeats.rename(columns={j:f"Feature_{i+1}" for i,j in enumerate(NeurFeats)}, inplace=True)
NeurFeats.head()

# Preprocessing and Mapping Output Labels

ClassValues = pd.read_csv('/Users/teresaibarra/Downloads/merged_labeled_DevAttentionY.csv')
ClassValues.drop(list(ClassValues.columns)[0], axis=1, inplace=True)
ClassValues.rename(columns={'2':'Attended/Unattended'}, inplace=True)
ClassValues.head()

# Working with first 120000 records in both datasets and scaling neural features

NeurFeatExp = NeurFeats.loc[0:119999]
ClassValuesExp = ClassValues.loc[0:119999]
ZNeurFeat = scale(NeurFeatExp, axis=0)

# Splitting into Training and Testing Sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(ZNeurFeat, ClassValuesExp, test_size=0.2, random_state=100)

# Splitting into Training and Validation Sets
Xtrain2, Xval, Ytrain2, Yval = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=100)

# Building a Logistic Regression Model

model = LogisticRegression(solver = 'lbfgs')
model_fit = model.fit(Xtrain2, Ytrain2)
test_pred = model_fit.predict(Xtest)
test_pred_prob = model_fit.predict_proba(Xtest)

# Accuracy Score and Plotting ROC Curve and Confusion Matrix
print(f"Accuracy of model is: {round(accuracy_score(Ytest, test_pred)*100, 2)}%")
plot_roc(Ytest, test_pred_prob, title="Logistic Regression ROC")
plot_confusion_matrix(Ytest, test_pred, title="Logistic Regression Confusion Matrix")

# Logistic Regression with L2 Penalty
log_model = LogisticRegression(penalty='l2', dual=True, solver='liblinear', random_state=100)
log_model_fit = log_model.fit(Xtrain2, Ytrain2)
log_model_pred = log_model_fit.predict(Xtest)
log_model_prob = log_model_fit.predict_proba(Xtest)

# Plot ROC and Confusion Matrix
print(f"Accuracy: {round(accuracy_score(Ytest, log_model_pred)*100,2)}%")
plot_roc(Ytest, log_model_prob, title="Logistic Regression Dual Formulation ROC")
plot_confusion_matrix(Ytest, log_model_pred, title="Logistic Regression Dual Formulation Confusion Matrix")

# Non-Dual Liblinear Logistic Regression Solver

log_model_2 = LogisticRegression(penalty='l2', dual=False, solver='liblinear', random_state=100)
log_model_2_fit = log_model_2.fit(Xtrain2,Ytrain2)
log_model_2_pred = log_model_fit.predict(Xtest)
log_model_2_prob = log_model_fit.predict_proba(Xtest)

# Plot ROC and Confusion Matrix
print(f"Accuracy: {round(accuracy_score(Ytest, log_model_2_pred)*100,2)}%")