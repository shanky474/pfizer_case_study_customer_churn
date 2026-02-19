from fastapi import FastAPI, Request
from typing import Any, Dict, List, Union
from utils import models
from utils import MLClassificationInference


app = FastAPI()



@app.post("/customerdetails")
def predict_values(customerdetails: List[models.CustomerDetails]):
    """
    Read the entire request body as a generic JSON structure.
    """

    inference = MLClassificationInference.Inference(r'artifacts')

    inference.load_artifact()

    customer_details_formatted = [{
                                    'gender': customerdetail.gender,
                                    'Partner': customerdetail.Partner,
                                    'Dependents': customerdetail.Dependents,
                                    'PhoneService':customerdetail.PhoneService,
                                    'MultipleLines':customerdetail.MultipleLines,
                                    'InternetService':customerdetail.MultipleLines,
                                    'OnlineSecurity':customerdetail.OnlineSecurity,
                                    'OnlineBackup':customerdetail.OnlineBackup,
                                    'DeviceProtection':customerdetail.DeviceProtection,
                                    'TechSupport':customerdetail.TechSupport,
                                    'StreamingTV':customerdetail.StreamingTV,
                                    'StreamingMovies':customerdetail.StreamingMovies,
                                    'Contract':customerdetail.Contract,
                                    'PaperlessBilling':customerdetail.PaperlessBilling,
                                    'PaymentMethod':customerdetail.PaymentMethod
                                    } for customerdetail in customerdetails]

    inference.preprocess_data(customer_details_formatted,['Churn'],['gender',
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
    # The .json() method is an awaitable that returns a Python data structure
    return {"predicted_Churn": predictions}