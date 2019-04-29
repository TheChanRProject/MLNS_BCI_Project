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
