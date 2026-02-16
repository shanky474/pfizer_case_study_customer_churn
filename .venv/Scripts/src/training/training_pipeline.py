from utils import PreProcessingClassificationChurn, FeatureExtractionClassificationChurn, MLTrainingClassificationChurn

data = PreProcessingClassificationChurn.PreProcessTrainingData(r'.venv\Scripts\data\raw\Telco-Customer-Churn.csv')

data.check_nulls()

data.check_duplicates()

data.pre_process_categorical_numeric_columns(['customerID'],['gender',
                                                             'Partner',
                                                             'Dependents',
                                                             'PhoneService',
                                                             'MultipleLines',
                                                             'InternetService',
                                                             'OnlineSecurity',
                                                             'OnlineBackup',
                                                             'DeviceProtection',
                                                             'TechSupport',
                                                             'StreamingTV',
                                                             'StreamingMovies',
                                                             'Contract',
                                                             'PaperlessBilling',
                                                             'PaymentMethod'],['tenure',
                                                                               'MonthlyCharges',
                                                                               'TotalCharges'],'Churn',r'.venv\Scripts\data\preprocessed\Telco-Customer-Churn-encoded.csv')



features = FeatureExtractionClassificationChurn.FeatureExtraction(r'.venv\Scripts\data\preprocessed\Telco-Customer-Churn-encoded.csv')
features.extract_relevant_features(['tenure','MonthlyCharges','TotalCharges'],r'.venv\Scripts\data\features\Telco-Customer-Churn-Features.csv')



model = MLTrainingClassificationChurn.TrainMLModel(r'.venv\Scripts\data\features\Telco-Customer-Churn-Features.csv')
print("Correcting Class Imbalance:")
model.correct_class_imbalance('Churn')
model.create_model_tracker(r'.venv\Scripts\src\tracking\tracker.csv')
model.train_test_split('Churn')
model.train_XGB_model(r'.venv\Scripts\tracking\tracker.csv')
model.save_model(r'.venv\Scripts\artifacts')