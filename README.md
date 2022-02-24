# Credit_Risk_Analysis

# Overview
In this project, we use Python to build and evaluate several machine learning models to predict credit risk using the credit card credit dataset from LendingClub, a peer-to-peer lending services company. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we will need to employ different techniques to train and evaluate models with unbalanced classes. 

We will:
* Oversample the data using the **RandomOverSampler** and **SMOTE** algorithms
* Undersample the data using the **ClusterCentroids** algorithm
* Use a combinatorial approach of over- and undersampling using the **SMOTEENN** algorithm. 
* Next, we will compare two new machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**, to predict credit risk.

 We will evaluate the performance of the models on whether they should be used to predict credit risk.
 
# Resources
* **Data Source:** [LoanStats_2019Q1.csv](https://github.com/ramya-ramamur/Credit_Risk_Analysis/tree/main/Resources)
* **Software:** Python 3.8.8, Pandas Dataframe, Jupyter Notebook 6.4.6, Anaconda Navigator 2.1.1, imbalanced-learn, skikit-learn
