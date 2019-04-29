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
classValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv").loc[0:119999]
