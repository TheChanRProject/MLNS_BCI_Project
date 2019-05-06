# Random Forest Results for First Classification Task
## Results Without Cross Validation

- ### Random Forest Model 1: Untuned with 1000 Decision Tree Estimators with Entropy Criterion
  - Accuracy: 99.17%
  - ROC Curve: ![](../../../images/random-forest/rf_entropy_ROC.png)
  - Confusion Matrix: ![](../../../images/random-forest/rf_entropy_matrix.png)
  - Tree Visualization: ![](../../../images/random-forest/random-forest-estimator-entropy.png)     

- ### Random Forest Model 2: Untuned with 1000 Decision Tree Estimators with Gini Index Criterion
  - Accuracy: 99.19%
  - ROC Curve: ![](../../../images/random-forest/rf_Gini_ROC.png)
  - Confusion Matrix: ![](../../../images/random-forest/rf_gini_matrix.png)
  - Tree Visualization: ![](../../../images/random-forest/random-forest-estimator-gini.png)

Best Cross-Validated Accuracy: 99.3%
Best Model Parameters: {'n_estimators': 800, 'max_depth': None, 'criterion': 'gini', 'bootstrap': True}

## Results With Cross Validation [Siena]

- ### Random Forest Model 1: Untuned with 1000 Decision Tree Estimators with Entropy Criterion
  - Accuracy: 99.12%
  - ROC Curve: ![](../../../images/random-forest/untuned-rf-roc-curve.png)
  - Confusion Matrix: ![](../../../images/random-forest/untuned-rf-confusion-matrix.png)
  - Tree Visualization: ![](../../../images/random-forest/untuned-random-forest-estimator.png)   

- ### Random Forest Model 2: Untuned with 1000 Decision Tree Estimators with Gini Index Criterion
  - Accuracy: 99.12%
  - ROC Curve: Same as Model 1
  - Confusion Matrix: Same as Model 1
  - Tree Visualization: Same as Model 1

## Results With Cross Validation [Rishov]

- ### Random Forest: Tuned with 1000 Trees, Gini Index, Bootstrap = False
  - Accuracy: 99.1%
  - Confusion Matrix: ![](../../../images/random-forest/intrasub-backup-confusion-matrix.png) 
