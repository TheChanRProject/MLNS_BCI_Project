# MLNS_BCI_Project
Final Project for NEUR182: "Machine Learning with Neural Signals". This BCI Project was built by [Rishov Chatterjee](https:github.com/TheChanRProject), [Teresa Ibarra](https://github.com/teresaibarra), [Siena Guerrero](https://github.com/sienaguerrero), and [SiKe Wang](https://github.com/sikewang98) using information and data from the following paper: ["Decoding auditory attention to instruments in polyphonic music using single-trial EEG classification."](https://www.ncbi.nlm.nih.gov/pubmed/24608228)

## 5 Parts

### Part 1: Binary Classification for Unattended versus Attended

Merged and Labeled DevAttentionX data and Merged DevAttentionY data available on the Google Drive.

Please Note: Make sure to drop the first column from the dataframe when reading the csv files otherwise you will get an error when you instantiate your classifier in scikit-learn.

### Intra-Subject File IDs for Google Colab

File ID for merged_labeled_DevAttentionX.csv in Google Colab: "1-s6kpsj5Gvc86FtIrfk_AhlRvVZRi8Mj"

File ID for merged_labeled_DevAttentionY.csv in Google Colab: "1tHrpcAJjUuerjXrDJ1RGNRCmffPtkW33"

### Cross Subject Code for Google Colab

```
p1_x = drive.CreateFile({'id':"1RoH6sXOdhaFk-P2BRQKIY9yiLbTevReJ"})
p1_x.GetContentFile("VPaan_DevAttentionX.csv")

p1_y = drive.CreateFile({'id': "1kfGHc9LHFHbVv2CMX_CqnXECiXshY9Eb"})
p1_y.GetContentFile("VPaan_DevAttentionY.csv")

p2_x = drive.CreateFile({'id': "1fbaIh9xAcZMO35gVoym6iTnpIJvc1DcF"})
p2_x.GetContentFile("VPaap_DevAttentionX.csv")

p2_y = drive.CreateFile({'id': "1c7P3RbnyWCkhRUu_bv_SjUiJPdwusXKd"})
p2_y.GetContentFile("VPaap_DevAttentionY.csv")

p3_x = drive.CreateFile({'id': "1_fZHmXRRtZWZDY_L92GV-_gqWZjyaZyK"})
p3_x.GetContentFile("VPaas_DevAttentionX.csv")

p3_y = drive.CreateFile({'id': "1R_5qOWIikz8VHN1TJIgl_Cv_Zeh8-Ava"})
p3_y.GetContentFile("VPaas_DevAttentionY.csv")

p4_x = drive.CreateFile({'id': "1-7L3M541OdGGm1U2QjXXUV9ity1FiEZl"})
p4_x.GetContentFile("VPgcc_DevAttentionX.csv")

p4_y = drive.CreateFile({'id': "1rsY0_5MtJ9W_9EfLCKhNjSPnvBoIHPC9"})
p4_y.GetContentFile("VPgcc_DevAttentionY.csv")


```


[Importing Files from Drive onto Google Colab](https://buomsoo-kim.github.io/python/2018/04/16/Importing-files-from-Google-Drive-in-Google-Colab.md/)

Please look at src/ml_experiments/logistic-regression/rishov-logistic-regression.py for the end to end template that is required for all the models.

## Best Model for Task 1: Random Forest

- Random Forest
  - [Results](https://github.com/TheChanRProject/MLNS_BCI_Project/blob/master/results/Unattended_Attended/random-forest/results.md)


- Logistic Regression
  - [Results](https://github.com/TheChanRProject/MLNS_BCI_Project/blob/master/results/Unattended_Attended/logistic-regression/results.md)


- Linear Discriminant Analysis
  - [Results](https://github.com/TheChanRProject/MLNS_BCI_Project/blob/master/results/Unattended_Attended/lda/results.md)

- Neural Network
  - [Results](https://github.com/TheChanRProject/MLNS_BCI_Project/blob/master/results/Unattended_Attended/neural-network/results.md)
  - To Do:
    - Visualize Tuned Neural Network 

- Naive Bayes
  - [Results](https://github.com/TheChanRProject/MLNS_BCI_Project/blob/master/results/Unattended_Attended/naive-bayes/results.md)
  - To Do
    1. Implement Stochastic Gradient Descent (SGD) Classifier with modified-huber loss function
    2. Use regularization: l1, l2, ElasticNet
    3. Implement cross validation
    4. If time available: Ensemble with AdaBoost  

### Part 2: Multi-Class Classification for Unattended versus Attended Including Instruments
- Multi-class LDA
- Neural Network
- Logistic Regression with Softmax
- Random Forest

### Part 3: Finding Out the Most Important Features

- Relevance Vector Machine [If Time Available]
- Random Forest Feature Importance
- Recursive Feature Elimination with Logistic Regression
- Stepwise Regression [If Time Available]

### Part 4: Putting it together in a paper

Resources:

[Classification of EEG data using machine learning techniques](http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8895013&fileOId=8895015)

### Part 5: Building the poster


## Tasks:

1. Build a random forest (Siena)

Tutorials to look at:

- [Will Koehrsen: Random Forest End to End](https://towardsdatascience.com/random-forest-in-python-24d0893d51c0)
- [Will Koehrsen: Visualizing the Random Forest](https://towardsdatascience.com/how-to-visualize-a-decision-tree-from-a-random-forest-in-python-using-scikit-learn-38ad2d75f21c)
- If Time Available: [Optimizing of Ensemble Classifiers using Genetic Algorithm](https://pdfs.semanticscholar.org/3ac5/fe864ef84b4b764f600ccf67c980d0e9ac94.pdf)

- [Feature Importance in Random Forests](https://towardsdatascience.com/running-random-forests-inspect-the-feature-importances-with-this-code-2b00dd72b92e)

2. Build a logistic regression (Teresa)

Tutorials to look at:

- [Susan Li: Building a Logistic Regression in Python](https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)
- [William Cohen: Logistic Regression with Stochastic Gradient Descent](http://www.cs.cmu.edu/~wcohen/10-605/sgd-part2.pdf)
- [Scikit-learn Docs: SGD Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)

3. Build a Neural Network (SiKe)

Tutorials to look at:

- [Neural Networks and Backprop in Scikitlearn](https://www.youtube.com/watch?v=X8SPO875mQY)
- [GitHub for the Above Video](https://github.com/shreyans29/thesemicolon/blob/master/Neural%20Networks%20and%20BackPropogation.ipynb)

- [Sklearn Neural Network API Docs](https://scikit-neuralnetwork.readthedocs.io/en/latest/module_mlp.html)

- [Visualizing Neural Net](https://towardsdatascience.com/visualizing-artificial-neural-networks-anns-with-just-one-line-of-code-b4233607209e)

4. Cross Validation

- [Resources for Grid Search](https://stackabuse.com/cross-validation-and-grid-search-for-model-selection-in-python/)
- [Resources for Grid Search](https://stats.stackexchange.com/questions/375682/difference-between-using-cv-5-or-cv-kfoldn-splits-5-in-cross-val-score)
- [Randomized vs Grid Search](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)
- [ROC for Cross Validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html)
