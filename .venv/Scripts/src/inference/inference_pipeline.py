from utils import MLClassificationInference
import json
import pandas as pd


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

for item_dict,predicted_value in zip(inference_data,predictions):
    item_dict.pop('Churn', None)
    item_dict['Predicted_Churn']=str(predicted_value)


final_predicted_data = pd.DataFrame(inference_data)

print("Saving Final Prediction results: ")

final_predicted_data.to_csv(r'.venv\Scripts\data\predictions\Telco-Customer-Churn-Predictions.csv',index=False)

# if __name__ == "__main__":
#     # Load model and preprocessor once
#     # model, preprocessor = load_artifact()
#     load_artifact()
#     print("Model and preprocessor loaded successfully.")