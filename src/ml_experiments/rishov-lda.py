import numpy as np, matplotlib.pyplot as plt, scipy, sklearn, statsmodels, parfit
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# *
ClassValues = pd.read_csv('data/Merged/merged_DevAttentionY.csv')
NeurFeat = pd.read_csv('data/Merged/merged_DevAttentionX.csv')

# Split into Train and Validation and Test Sets
# Center and Scale
ZNeurFeat = scale(NeurFeat,axis=0)
XTrain, XTest, YTrain, YTest = train_test_split(
        ZNeurFeat,ClassValues,test_size = 0.2,
        random_state = 100)

# Split Train Set into Train and Validation Sets
XTrain2, XVal, YTrain2, YVal = train_test_split(
        XTrain,YTrain,test_size = 0.2,
        random_state = 100)

# *
# Recover Classes Using Least Squares Linear Discriminant Analysis
LDAMod = LinearDiscriminantAnalysis(solver = 'lsqr', shrinkage='auto')
LDAFit = LDAMod.fit(XTrain2,YTrain2)
TestPred = LDAFit.predict_proba(XTest)
TestPred2 = LDAFit.predict(XTest)
plot_roc(YTest,TestPred,title = 'LDA')
plot_confusion_matrix(YTest,TestPred2)
print(accuracy_score(YTest, TestPred2))
# *
