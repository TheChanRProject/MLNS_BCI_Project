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

# Reading in Features from Labeled DevAttentionX Data with first 120000 records
neuralFeatures = pd.read_csv("data/Merged/merged_labeled_DevAttentionX.csv").loc[0:119999]
# Reading in Outputs: Attended or Unattended from Labeled DevAttentionY Data with first 120000
classValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY.csv").loc[0:119999]
