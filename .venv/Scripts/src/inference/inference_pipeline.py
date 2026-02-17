from utils import MLClassificationInference
import json


with open(r'.venv\Scripts\data\data.json', "r") as file:
    inference_data = json.load(file)

inference = MLClassificationInference.Inference(r'.venv\Scripts\artifacts')

inference.load_artifact()

inference.preprocess_data(inference_data,['Churn'],['gender',
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
                                            'PaymentMethod'],['tenure','MonthlyCharges','TotalCharges'])

predictions = inference.predict({1:"Yes",0:"No"})

print(predictions)
# if __name__ == "__main__":
#     # Load model and preprocessor once
#     # model, preprocessor = load_artifact()
#     load_artifact()
#     print("Model and preprocessor loaded successfully.")