import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc, plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
from warnings import filterwarnings
filterwarnings("ignore")

# X File Read in Pandas
NeurFeat = pd.read_csv("data/Merged/merged_labeled_DevAttentionX_v2.csv")
NeurFeat.drop(list(NeurFeat.columns)[0], axis=1, inplace=True)
print(NeurFeat.shape)

# Y File Read in Pandas
ClassValues = pd.read_csv("data/Merged/merged_labeled_DevAttentionY_v2.csv")
ClassValues.drop(list(ClassValues.columns)[0], axis=1, inplace=True)
print(ClassValues.shape)

# Scale neural features
zFeat = scale(NeurFeat, axis=0)

# Split data into training and testing sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(zFeat, ClassValues, test_size=0.2, random_state=100)

# Partition training set into training and validation sets
Xtrain2, Xval, Ytrain2, Yval = train_test_split(Xtrain, Ytrain, test_size=0.2, random_state=100)

# Random Forest with Gini Index
rf_model_gini = RandomForestClassifier(n_estimators=1000, criterion='gini')
rf_fit_gini = rf_model_gini.fit(Xtrain2, Ytrain2)
rf_pred_gini = rf_fit_gini.predict(Xtest)
rf_prob_gini = rf_fit_gini.predict_proba(Xtest)

# Random Forest with Entropy
rf_model_entropy = RandomForestClassifier(n_estimators=1000, criterion='entropy')
rf_fit_entropy = rf_model_entropy.fit(Xtrain2, Ytrain2)
rf_pred_entropy = rf_fit_entropy.predict(Xtest)
rf_prob_entropy = rf_fit_entropy.predict_proba(Xtest)

# Plot ROC and Confusion Matrix for Gini
print(f"Accuracy Score: {round(accuracy_score(Ytest, rf_pred_gini)*100,2)}%")
plot_roc(Ytest, rf_prob_gini, title="Random Forest with Gini Index ROC")
plot_confusion_matrix(Ytest, rf_pred_gini, title="Random Forest with Gini Index Confusion Matrix")

# Plot ROC and Confusion Matrix for Entropy
print(f"Accuracy Score: {round(accuracy_score(Ytest, rf_pred_entropy)*100,2)}%")
plot_roc(Ytest, rf_prob_entropy, title="Random Forest with Entropy ROC")
plot_confusion_matrix(Ytest, rf_pred_entropy, title="Random Forest with Entropy Confusion Matrix")

# Cross Validation Function for Random Forest
def RandomForest(x_df, y_df, cv, n):
    model = RandomForestClassifier(random_state=100, verbose=True)
    params = {"n_estimators": [200, 400, 600, 800, 1000],
              "max_depth": [3, None],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
    random_search = RandomizedSearchCV(model, param_distributions=params, cv=cv, n_iter=n)
    random_fit = random_search.fit(x_df, y_df)
    print(random_fit.cv_results_)
    print(f"Best Cross-Validated Accuracy: {round(random_fit.best_score_*100,2)}%")
    print(f"Best Model Parameters: {random_fit.best_params_}")
    return random_fit

RandomForest(zFeat, ClassValues, 5, 2) #Run Randomized Search Cross Validation with Random Forest

## Visualizing trees used to estimate the random forest for Gini and Entropy ##
from sklearn.tree import export_graphviz
import pydot

# Visualizing a single tree estimator
tree_gini = rf_model_gini.estimators_[5]
tree_entropy = rf_model_entropy.estimators_[5]

# Export image
export_graphviz(tree_gini, out_file="images/random-forest-estimator_gini.dot", feature_names=list(NeurFeat.columns), rounded=True, precision=1)
export_graphviz(tree_entropy, out_file="images/random-forest-estimator_entropy.dot", feature_names=list(NeurFeat.columns), rounded=True, precision=1)

# Use dot file to create a graph
(graph_gini, ) = pydot.graph_from_dot_file('images/random-forest-estimator_gini.dot')
(graph_entropy, ) = pydot.graph_from_dot_file('images/random-forest-estimator_entropy.dot')

# Write graph to a png file
graph_gini.write_png('random-forest-estimator-gini.png')
graph_entropy.write_png('random-forest-estimator-entropy.png')

## Feature Importance ##

# Feature Importance for Gini
feature_importances_gini = pd.DataFrame(rf_fit_gini.feature_importances_, index = NeurFeat.columns, columns=['importance']).sort_values('importance', ascending=False)

# Feature Importance for Entropy
feature_importances_entropy = pd.DataFrame(rf_fit_entropy.feature_importances_, index = NeurFeat.columns, columns=['importance']).sort_values('importance', ascending=False)

print(feature_importances_gini)
print(feature_importances_entropy)

feature_importances_gini.to_csv(“data/Feature_Importance/rf_gini_features.csv”)
feature_importances_entropy.to_csv(“data/Feature_Importance/rf_entropy_features.csv”)


