import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
filterwarnings("ignore")
from jupyterthemes import jtplot
jtplot.style(theme='monokai')

NeurFeat = pd.read_csv("data/Merged/merged_labeled_DevAttentionX.csv").loc[0:119999]
ClassValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv").loc[0:119999]
NeurFeat.drop(list(NeurFeat.columns)[0], axis=1, inplace=True)
ClassValues.drop(list(ClassValues.columns)[0], axis=1, inplace=True)
zFeat = scale(NeurFeat, axis=0)

# Training and Testing Sets
Xtrain1, Xtest, Ytrain1, Ytest = train_test_split(zFeat, ClassValues, test_size=0.2, random_state=100)
# Training -> Training and Validation Sets
Xtrain2, Xval, Ytrain2, Yval = train_test_split(Xtrain1, Ytrain1, test_size=0.2, random_state=100)
