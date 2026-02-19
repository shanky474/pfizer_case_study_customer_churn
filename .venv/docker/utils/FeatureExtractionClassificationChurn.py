import pandas as pd

class FeatureExtraction():
    def __init__(self,pre_processed_data_path):
        self.data = pd.read_csv(f"{pre_processed_data_path}",index_col=None)

    def extract_relevant_features(self,columns_to_drop,features_save_path):
        print('''**Final Observations**


                    *   Churn by OnlineSecurity Type shows that abscence of security results in higher churn. We see similar outcomes for absence of OnlineBackup so we pick both.

                    *   Paperless Billing type results in higler churn which is correlated to PaymentType electronic check hence we would pick both.

                    *   In our previous observations itself we have found that Month-to-month Contract type is directly correlated to churn hence we would pick this feature.

                    *   In our previous observations itself we have found that abscence of Tech support is directly correlated to churn hence we would pick this feature.

                    *   As we see that there is a marked class imbalance skewed in favour of no churn hence we have to use SMOTE to balance it out.

                    *   Also we see that tenure, MonthlyCharges, TotalCharges are highly correlated to the above categorical features (Contract, TechSupport, PaperlessBilling, OnlineSecurity) hence we are going to drop these features.

                    *   There is a marked class imbalance skewed in favour of no churn hence we have to use SMOTE to balance it out.''')
        
        self.data.drop(columns_to_drop,axis=1,inplace=True)
        self.data.to_csv(f"{features_save_path}",index=False)