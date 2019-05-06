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
  - [Intra-Subject Results](results/Unattended_Attended/Intra-Subject/random-forest/results.md)
  - [Cross-Subject Results](results/Unattended_Attended/Cross-Subject/random-forest/results.md)


- Logistic Regression
  - [Intra-Subject Results](results/Unattended_Attended/Intra-Subject/logistic-regression/results.md)


- Linear Discriminant Analysis
  - [Intra-Subject Results](results/Unattended_Attended/Intra-Subject/lda/results.md)
  - [Cross-Subject Results](results/Unattended_Attended/Intra-Subject/logistic-regression/results.md) 
- Neural Network
  - [Intra-Subject Results](results/Unattended_Attended/Intra-Subject/neural-network/results.md)
  - [Cross-Subject Results]()

- Naive Bayes
  - [Results](results/Unattended_Attended/Intra-Subject/naive-bayes/results.md)


### Part 2: Multi-Class Classification for Unattended versus Attended Including Instruments [Save for Paper]
- Multi-class LDA
- Neural Network
- Logistic Regression with Softmax
- Random Forest


### Part 4: Putting it Together for  Final Paper

Resources:

[Classification of EEG data using machine learning techniques](http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8895013&fileOId=8895015)

### Part 5: Building the poster

https://docs.google.com/document/d/1ZBoO8Kj0ctLfLiV6ddr7w1Xk3D0tTq8KQJe2ELYCMyw/edit?usp=sharing
