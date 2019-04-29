import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from warnings import filterwarnings
filterwarnings("ignore")

# Reading in Features from Labeled DevAttentionX Data with first 120000 rows
neuralFeatures = pd.read_csv("data/Merged/merged_labeled_DevAttentionX.csv").loc[0:119999]
# Drop Unnamed column
neuralFeatures.drop(list(neuralFeatures.columns)[0], axis=1, inplace=True)
# Reading in Outputs: Attended or Unattended from Labeled DevAttentionY Data
classValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv").loc[0:119999]
# Drop Unnamed column
classValues.drop(list(classValues.columns)[0], axis=1, inplace=True)
# Scaling features
zFeatures = scale(neuralFeatures, axis=0)
# Creating Training and Testing Sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(zFeatures, classValues, test_size=0.2, random_state=100)

# Partitioning Training Set into Training and Validation Sets
Xtrain2, Xval, Ytrain2, Yval = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=100)

# Building a Naive Bayes Classifier Assuming a Gaussian Distribution
nb_model = GaussianNB()
nb_fit = nb_model.fit(Xtrain2, Ytrain2)
nb_pred_test = nb_fit.predict(Xtest)
nb_pred_test_prob = nb_fit.predict_proba(Xtest)

# Plotting ROC Curve and Confusion Matrix
from jupyterthemes import jtplot
jtplot.style(theme="monokai")
plot_roc(Ytest, nb_pred_test_prob, title="Naive Bayes ROC on Test Set")
