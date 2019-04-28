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

NeurFeats = pd.read_csv("data/Merged/merged_DevAttentionX.csv")
NeurFeats.drop(list(NeurFeats.columns)[0], axis=1, inplace=True)
NeurFeats.rename(columns={j:f"Feature_{i+1}" for i,j in enumerate(NeurFeats)}, inplace=True)
NeurFeats.head()
