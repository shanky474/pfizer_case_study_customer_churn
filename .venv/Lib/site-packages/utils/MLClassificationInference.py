import joblib
import pandas as pd
import numpy as np
import os


class Inference():
    def __init__(self,artifacts_path):
        print(artifacts_path)
        self.artifacts = os.listdir(rf"{artifacts_path}")
        self.artifact_path = artifacts_path


    def load_artifact(self):
        """Loads the trained model and preprocessor from disk."""
        self.model = joblib.load(os.path.join(self.artifact_path, self.artifacts[-1]))

    
    def preprocess_data(self,data,drop_columns,categorical_columns,numeric_columns):
        self.data = pd.DataFrame(data)
        self.data.drop(drop_columns,axis=1,inplace=True, errors='ignore')
        self.data_fine_tune_encoded = pd.get_dummies(self.data, columns=categorical_columns,drop_first=False,dtype='int')
        self.data_fine_tune_encoded.columns = self.data_fine_tune_encoded.columns.str.replace(' ', '_')
        self.data_fine_tune_encoded.drop(numeric_columns,axis=1, inplace=True, errors='ignore')
        self.prediction_columns = ['SeniorCitizen', 'gender_Female', 'gender_Male', 'Partner_No',
       'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No',
       'PhoneService_Yes', 'MultipleLines_No',
       'MultipleLines_No_phone_service', 'MultipleLines_Yes',
       'InternetService_DSL', 'InternetService_Fiber_optic',
       'InternetService_No', 'OnlineSecurity_No',
       'OnlineSecurity_No_internet_service', 'OnlineSecurity_Yes',
       'OnlineBackup_No', 'OnlineBackup_No_internet_service',
       'OnlineBackup_Yes', 'DeviceProtection_No',
       'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
       'TechSupport_No', 'TechSupport_No_internet_service', 'TechSupport_Yes',
       'StreamingTV_No', 'StreamingTV_No_internet_service', 'StreamingTV_Yes',
       'StreamingMovies_No', 'StreamingMovies_No_internet_service',
       'StreamingMovies_Yes', 'Contract_Month-to-month', 'Contract_One_year',
       'Contract_Two_year', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
       'PaymentMethod_Bank_transfer_(automatic)',
       'PaymentMethod_Credit_card_(automatic)',
       'PaymentMethod_Electronic_check', 'PaymentMethod_Mailed_check']
        self.current_columns = self.data_fine_tune_encoded.columns
        self.data_fine_tune_encoded_matched = pd.DataFrame([[0 for col in self.prediction_columns] for i in range(len(self.data_fine_tune_encoded))],columns=self.prediction_columns)
        self.data_fine_tune_encoded_matched.update(self.data_fine_tune_encoded)
        # for column in self.prediction_columns:
        #     if column not in self.current_columns:
        #         print(column)
        #     else:
        #         [column]=0
        

    def predict(self,label_mapping):
        
        # print(self.data_fine_tune_encoded)
        self.model_predictions = self.model.predict(self.data_fine_tune_encoded_matched)
        self.map_values = np.vectorize(label_mapping.get)
        # print(self.map_values(self.model_predictions))
        return list(self.map_values(self.model_predictions))
        # return self.data_fine_tune_encoded_matched.to_dict(orient='records')[0]


