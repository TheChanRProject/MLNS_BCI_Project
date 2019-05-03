import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# Test with function from randomized search
def cv_roc(n, fit_function, Ytest, Xtest):
    probas = fit_function.predict_proba(Xtest)
