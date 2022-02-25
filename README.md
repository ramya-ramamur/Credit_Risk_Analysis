Determine the best machine learning model to predict credit risk: **NaiveRandomOverSampler, SMOTE, SMOTEENN, BalancedRandomForestClassifier and EasyEnsembleClassifier.**

# Credit_Risk_Analysis

# Overview
In this project, we use Python to build and evaluate several machine learning models to predict credit risk using the credit card credit dataset from LendingClub, a peer-to-peer lending services company. Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, we will need to employ different techniques to train and evaluate models with unbalanced classes. 

We will:
* Oversample the data using the **NaiveRandomOverSampler** and **SMOTE** algorithms
* Undersample the data using the **ClusterCentroids** algorithm
* Use a combinatorial approach of over- and undersampling using the **SMOTEENN** algorithm. 
* Next, we will compare two new machine learning models that reduce bias, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier**, to predict credit risk.

 We will evaluate the performance of the models on whether they should be used to predict credit risk.
 
# Resources
* **Data Source:** [LoanStats_2019Q1.csv](https://github.com/ramya-ramamur/Credit_Risk_Analysis/tree/main/Resources)
* **Software:** Python 3.8.8, Pandas Dataframe, Jupyter Notebook 6.4.6, Anaconda Navigator 2.1.1, imbalanced-learn, skikit-learn

# Results

**NaiveRandomOverSampler Model**

Naïve random oversampling allows generating new samples by randomly sampling the current samples of the minority class with replacement. 
In random oversampling, instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced. An instance of RandomOverSampler is instantiated. The training data (X_train and y_train) is resampled using the fit_resample() method. The results are called X_resampled and y_resampled. Counting the classes of the resampled target verifies that the minority class has been enlarged.

* The balanced accuracy score is ~64%.
* The high_risk precision is ~1% only with 63% sensitivity and a F1 of 2%.
* Precision is almost 100% with a sensitivity of 66% for the low_risk population.

<img width="1119" alt="Naive Random Oversampling" src="https://user-images.githubusercontent.com/75961057/155653168-4141db88-148c-4e8f-b40c-ddc88356a7d5.png">

**SMOTE model**

The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

This model did not return any scores better than NaiveRandomOverSampler Model. 
* The balanced accuracy score is still ~65%.
* The high_risk precision is ~1% only with 62% sensitivity and a F1 of 2%.
* Precision is almost 100% with a sensitivity of 68% for the low_risk population.

<img width="1135" alt="SMOTE" src="https://user-images.githubusercontent.com/75961057/155653226-9683e82a-8d60-4d6e-96ce-300539125829.png">

**Cluster Centriods**

This technique makes undersampling by generating a new set based on centroids by clustering methods.  Undersamples the majority class by replacing a cluster of majority samples by the cluster centroid of a KMeans algorithm. This algorithm keeps N majority samples by fitting the KMeans algorithm with N cluster to the majority class and using the coordinates of the N cluster centroids as the new majority samples.

This model performed worse than the NaiveRandomOverSampler Model and Cluster Centriods. 
* The balanced accuracy score is ~52%.
* The high_risk precision is ~1% only with 63% sensitivity and a F1 of 1%.
* Precision is almost 100% with a sensitivity of 40% for the low_risk population.

<img width="1139" alt="Cluster Centroids" src="https://user-images.githubusercontent.com/75961057/155653271-029de5e5-9577-4451-b8c2-e1b962a99997.png">

**SMOTEENN**

SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:
Oversample the minority class with SMOTE. Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

This model performed better than Cluster Centroids.
* The balanced accuracy score is ~63%.
* The high_risk precision is ~1% only with 68% sensitivity and a F1 of 2%.
* Precision is almost 100% with a sensitivity of 57% for the low_risk population.

<img width="1136" alt="SMOTEENN" src="https://user-images.githubusercontent.com/75961057/155653303-75d77abf-a7f0-44ee-8fdf-0e624cf0b7ba.png">

**BalancedRandomForestClassifier**

BalancedRandomForestClassifier is an ensemble method in which each tree of the forest will be provided a balanced bootstrap sample. 

* The balanced accuracy score improved to about 77%.
* The high_risk precision is still low at 4% only with 67% sensitivity which makes a F1 of only 7%.
* Due to a lower number of false positives, the low_risk sensitivity is now 91% with 100% presicion.

<img width="1140" alt="Balanced Random Forest Classifier" src="https://user-images.githubusercontent.com/75961057/155653351-941cbfa3-c43c-49b7-b737-b56c985cf30d.png">

**EasyEnsembleAdaBoostClassifier**

A specific method which uses AdaBoostClassifier as learners in the bagging classifier is called “EasyEnsemble”. The EasyEnsembleClassifier allows to bag AdaBoost learners which are trained on balanced bootstrap samples. 
This model performed the best among the 6 models. 
* The balanced accuracy score is high to about 93%.
* The high_risk precision is still low at 7% only with 93% sensitivity which makes a F1 of only 13%.
* The low_risk sensitivity is now 93% with 100% presicion.

<img width="1143" alt="Easy Ensemble Ada Boost Classifier" src="https://user-images.githubusercontent.com/75961057/155653454-6e3b8e2d-519e-4a59-9da5-7ccc222d917a.png">

# Summary
We can see from summary statistics for the models, both Ensemble Classifiers outperformed the Resampling techniques in accurately predicted high-risk credit card applicants and low-risk applicants. However, all the models used to perform the credit risk analysis show weak precision in determining if a credit risk is high. The EasyEnsembleClassifier model shows a recall of 93% so it detects almost all high risk credit. On another hand, with a low precision, a lot of low risk credits are still falsely detected as high risk. I cannot recommend any of the models tested in this project for implementation as the lending institution risks losing ~90% of future customers and revenues of low-risk applicants when they are rejected for a credit-card. 




