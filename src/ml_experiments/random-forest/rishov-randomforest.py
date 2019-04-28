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

# X File Read in Pandas
NeurFeat = pd.read_csv("data/Merged/merged_DevAttentionX.csv")
NeurFeat.drop(list(NeurFeat.columns)[0], axis=1, inplace=True)
print(NeurFeat.shape)

# Y File Read in Pandas
ClassValues = pd.read_csv("data/Merged/merged_DevAttentionY.csv")
ClassValues.drop(list(ClassValues.columns)[0], axis=1, inplace=True)
print(ClassValues.shape)

# Getting the first 120000 records for both datasets

NeurFeatExp = NeurFeat.loc[0:119999]
ClassValuesExp = ClassValues.loc[0:119999]

ZFeatures = scale(NeurFeatExp, axis=0)
# Creating Training and Testing Sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(ZFeatures, ClassValuesExp, test_size=0.2, random_state=100)

# Create the validation sets
Xtrain2, XVal, Ytrain2, YVal = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=100)

# Building the Random Forest

model = RandomForestClassifier(n_estimators=1000, criterion="entropy")
model_fit = model.fit(Xtrain2, Ytrain2)
test_pred = model_fit.predict(Xtest)
test_pred_prob = model_fit.predict_proba(Xtest)
print(accuracy_score(Ytest, test_pred))

# Plotting the ROC Curve and Confusion Matrix 

from jupyterthemes import jtplot

jtplot.style(theme='monokai')
plt.figure(figsize=(12,8))
plot_roc(Ytest, test_pred_prob, title="Untuned Random Forest ROC Curve")
plt.savefig('images/rf-untuned-roc-curve.png')


plot_confusion_matrix(Ytest, test_pred, title="Untuned Random Forest Confusion Matrix")
plt.savefig('images/rf-untuned-confusion-matrix.png')

from sklearn.tree import export_graphviz
import pydot

# Visualizing a single tree estimator
tree = model.estimators_[5]

# Export image
export_graphviz(tree, out_file="images/untuned-random-forest-estimator.dot", feature_names=list(NeurFeatExp.columns), rounded=True, precision=1)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('images/untuned-random-forest-estimator.dot')
# Write graph to a png file
graph.write_png('untuned-random-forest-estimator.png')
