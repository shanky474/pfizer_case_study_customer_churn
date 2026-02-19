# Pfizer Case Study Customer Churn
This contains the source code for building ML model to predict customer churn as a case study assessment.


<!-- ABOUT THE PROJECT -->
## Overview

Curated codebase to build a model to predict customer churn based on customer demographics, subscription_details, tenure details. The code should perform following processes to generate artifacts and perform predictions: 

1. Use notebooks to preprocess the data for nulls and duplicates, explore the data to understand the features, categorize them into categorical, numeric features. 

2. Analyze these features (EDA) to find insights and relationships between them and categorize them into dependent as well as independent features with requisite dependencies and correlations. 

3. Avoid multicollinearity by using Variable Inflation Factor (VIF) techniques and Pearsons correlation.

4. Select only relevant features for the next step.

5. Perform Feature Engineering on these features using One Hot Encoding, Label Encoding techniques as desired for categorical features and scaling for numeric features. We can also impute missing features or drop missing rows as per the count.

6. Split featues into train test and use SMOTE techniques for upsampling. Train models like Logistic Regression, Decision tree, RandonForest, XGBoost, AdaBoost, LGBoost using these features and evaluate based on metrics like Accuracy, Precision, Recall, F1 score.

7. With Accuracy as scoring metric we pick the best model out of all models and use FPR, TPR thresholds to plot ROC Curve to do so. 

8. We create a training pipline to train the model incrementally and analyze it's performance metrics based on new unseen batches of data.

9. For real-time inference we package this model as an artifact and containerize it into a docker image with FASTAPI used to render the model.



<!-- ABOUT THE PROJECT -->
## Codebase Architecture

Please refer the following architecture diagram to focus on the processes implementation above

```
----.venv
       |----Scripts
                |----notebooks
                            |----Customer_Churn_EDA.ipynb
                            |----ML_Modelling.ipynb
                |----src
                        |----inference
                                    |----inference_pipeline.py
                                    |----main.py
                        |----training
                                    |----training_pipeline.py
                |----tracking
                            |----tracker.csv
                |----data
                        |----raw
                                |----Telco-Customer-Churn.csv
                        |----preprocessed
                                |----Telco-Customer-Churn-encoded.csv
                        |----features
                                |----Telco-Customer-Churn-Features.csv
                        

```