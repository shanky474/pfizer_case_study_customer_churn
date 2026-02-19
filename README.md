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



<!-- PROJECT STRUCTURE -->
## Architecture

### Codebase Architecture
Please refer the following codebase architecture diagram to focus on the modules/files to implement the project

```
|----.venv
       |----Lib\site-packages
                            |----utils
                                     |----PreProcessingClassificationChurn.py
                                     |----FeatureExtractionClassificationChurn.py
                                     |----MLTrainingClassificationChurn.py
                                     |----MLClassificationInference.py
                                     |----models.py
       |----Scripts
                |----requirements.txt
                |----requirements.txt
                |----artifacts
                            |----XGB_model_2026-02-16_20_22_20.987057.joblib
                            |----XGB_model_2026-02-16_20_23_45.338177.joblib
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
                        |----features
                                |----Telco-Customer-Churn-Predictions.csv
                        |----data.json                        

```

### System Architecture
Please refer the following implementation architecture to understand various system stages to process the data, extract features, train on those features and build and evaluate models.


![AWS Architecture Rendition](.venv/Diagrams\pfizer_case_study_system_architecture.jpg)


<!-- DEEP DIVE -->
# Detailed Illustration

1. Notebooks `Customer_Churn_EDA.ipynb`, `ML_Modelling.ipynb` are used to perform analysis on preprocessing, EDA, Feature Extraction, Feature Engineering, ML Model Development for exploration purposes.

2. Based on the above findings `training_pipeline.py` trains the selected model incrementally and stores the artifacts.

3. Post training `inference_pipeline.py` performs batch inference on the latest model in artifacts by sourcing batched data `data.json`

4. Real-time predictions are done by FASTAPI based framework invoked by `main.py`. (Run command `uvicorn main:app --reload`. Remove --reload for production deployment) 

4. Model and inference scripts are packaged into a docker container. Copy files/directories `main.py`, `artifacts`, `utils` and generate `Dockerfile` into the following folder structure. 

```
|----docker
        |----artifacts
        |----utils
        |----Dockerfile
        |----main.py
        |----requirements.txt

```

[def]: Diagrams\pfizer_case_study_system_architecture.png