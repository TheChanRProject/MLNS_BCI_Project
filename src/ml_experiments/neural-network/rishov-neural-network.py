import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from warnings import filterwarnings
filterwarnings("ignore")

# Read in labeled neural features
neuralFeatures = pd.read_csv("data/Merged/merged_labeled_DevAttentionX.csv").loc[0:119999]
neuralFeatures.drop(list(neuralFeatures.columns)[0], axis=1, inplace=True)

# Read in labeled outputs Attended versus Unattended
classValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv").loc[0:119999]
classValues.drop(list(classValues.columns)[0], axis=1, inplace=True)

# Scale neural features
zFeatures = scale(neuralFeatures, axis=0)

# Split data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(zFeatures, classValues, test_size=0.2, random_state=100)

# Partition training set into training and validation sets
Xtrain2, Xval, Ytrain2, Yval = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=100)

# Build multi-layer perceptron (Artificial Neural Network)
nn_model = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(10,15), random_state=1)
nn_fit = nn_model.fit(Xtrain2, Ytrain2)
nn_pred_test = nn_fit.predict(Xtest)
nn_pred_test_prob = nn_fit.predict_proba(Xtest)
