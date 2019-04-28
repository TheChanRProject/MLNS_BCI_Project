import numpy as np, matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from warnings import filterwarnings
filterwarnings("ignore")

# Preprocessing datasets

ClassValues = pd.read_csv('data/Merged/merged_DevAttentionY.csv')
NeurFeat = pd.read_csv('data/Merged/merged_DevAttentionX.csv')

NeurFeat.drop(list(NeurFeat.columns)[0], axis=1, inplace=True)

NeurFeat.head()
ClassValues.drop(list(ClassValues.columns)[0], axis=1, inplace=True)
ClassValues.head()

# Getting the first 120000 samples

NeurFeatExp = NeurFeat.loc[0:119999]
ClassValuesExp = ClassValues.loc[0:119999]
# Split into Train and Validation and Test Sets
# Center and Scale
ZNeurFeat = scale(NeurFeatExp,axis=0)
print(len(ZNeurFeat))
print(len(ClassValuesExp))

#  Splitting into training and testing sets 

XTrain, XTest, YTrain, YTest = train_test_split(ZNeurFeat,ClassValuesExp,test_size = 0.2,random_state = 100)
print(f"The shape of XTrain is {XTrain.shape}")
print(f"The shape of YTrain is {YTrain.shape}")
print(f"The shape of XTest is {XTest.shape}")
print(f"The shape of YTest is {YTest.shape}")
# Split Train Set into Train and Validation Sets
XTrain2, XVal, YTrain2, YVal = train_test_split(XTrain,YTrain,test_size = 0.2,random_state = 100)
print(f"The shape of XTrain is {XTrain2.shape}")
print(f"The shape of YTrain is {YTrain2.shape}")
print(f"The shape of XTest is {XVal.shape}")
print(f"The shape of YTest is {YVal.shape}")
# *
# Recover Classes Using Least Squares Linear Discriminant Analysis
LDAMod = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage='auto')
LDAFit = LDAMod.fit(XTrain2,YTrain2)
TestPred = LDAFit.predict_proba(XTest)
TestPred2 = LDAFit.predict(XTest)
from jupyterthemes import jtplot
jtplot.style(theme='monokai')
plot_roc(YTest,TestPred,title = 'LDA')
plot_confusion_matrix(YTest,TestPred2, title='LDA Confusion Matrix')
print(accuracy_score(YTest, TestPred2))
