# MLNS_BCI_Project
Final Project for Machine Learning with Neural Signals.

## 5 Parts

### Part 1: Binary Classification for Unattended versus Attended

- Random Forest
  - Results (Untuned Random Forest with Entropy)
    1. Accuracy: 99.1%
    2. ROC Curve: ![](images/random-forest/untuned-rf-roc-curve.png)
    3. Confusion Matrix: ![](images/random-forest/untuned-rf-confusion-matrix.png)
    4. Estimator Used to Train Random Forest Visualization: ![](images/untuned-random-forest-estimator.png)  
  - To Do
    1. Implement Cross Validation to Select Best Tree Estimator
    2. Visualize Best Tree Estimator
    3. Redo labels in Neural Features based on Paper 
- Logistic Regression
- LDA
- Neural Network
- Naive Bayes

### Part 2: Multi-Class Classification for Unattended versus Attended Including Instruments
- Multiclass LDA
- Neural Network
- Logistic Regression with Softmax

### Part 3: Finding Out the Most Important Features

- Relevance Vector Machine
- XGBoost

### Part 4: Putting it together in a paper

### Part 5: Building the poster


# Tasks:

1. Build a random forest

Tutorials to look at:

- [Will Koehrsen: Random Forest End to End](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)
- [Will Koehrsen: Visualizing the Random Forest](https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c)

2. Build a logistic regression

Tutorials to look at:

- [Susan Li: Building a Logistic Regression in Python](https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)

3. Build a Naive Bayes model

Tutorials to look at:

- [Gaurav Chauhan: All About Naive Bayes](https://towardsdatascience.com/all-about-naive-bayes-8e13cef044cf)
- [Naive Bayes with Scikit-Learn](https://github.com/2796gaurav/Naive-bayes-explained/blob/master/Naive%20bayes/Naive%20Bayes%20in%20scikit%20learn.ipynb)

4. Build a Neural Network

Tutorials to look at:

- [Neural Networks and Backprop in Scikitlearn](https://www.youtube.com/watch?v=X8SPO875mQY)
- [GitHub for the Above Video](https://github.com/shreyans29/thesemicolon/blob/master/Neural%20Networks%20and%20BackPropogation.ipynb)
