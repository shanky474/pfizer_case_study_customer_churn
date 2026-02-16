from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import datetime
import joblib


class TrainMLModel():
    def __init__(self,feature_data_path):
        self.data_encoded = pd.read_csv(f"{feature_data_path}",index_col=None)
        print(self.data_encoded.head())
        
        
    def correct_class_imbalance(self,target_column):
        max_value_class = self.data_encoded[target_column].value_counts().idxmax()
        max_value=self.data_encoded[target_column].value_counts().max()
        # Separate majority and minority classes
        self.data_encoded_majority = self.data_encoded[self.data_encoded[target_column]==max_value_class]
        self.data_encoded_minority = self.data_encoded[self.data_encoded[target_column]==1]

        # Upsample minority class
        self.data_encoded_minority_upsampled = resample(self.data_encoded_minority,
                                        replace=True,     # sample with replacement
                                        n_samples=	max_value,    # to match majority class
                                        random_state=123) # reproducible results

        # Combine majority class with upsampled minority class
        self.data_encoded_upsampled = pd.concat([self.data_encoded_majority, self.data_encoded_minority_upsampled])

        # Display new class counts
        print(self.data_encoded_upsampled[target_column].value_counts())


    def create_model_tracker(self,tracker_path):
        self.tracker = pd.DataFrame(columns=['Model','accuracy','precision','recall','f1_score','best_params','roc_auc_score','timestamp'])
        


    def train_test_split(self,target_column):
        X = self.data_encoded_upsampled.drop([target_column], axis=1)
        Y = self.data_encoded_upsampled[target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.3, random_state=101, stratify=Y)


    def train_XGB_model(self,tracker_path):
        print('''Observations:

                    1. We see that ROC curve clearly outlines the performance charecteristics of all the models. AUC scores of XGB, LGBM are the highest and similar hence we selct XGB as our predictor model

                    2. For almost all the models for which we could plot the feature contributions/weights we see that Contract month to month basis prominently features as a factor directly influencing the Churn. This corroborates with out earlier exploration finding as well during EDA.

                    3. Also predicted by logistic regression, 2 month contract has the most negative relation with Churn as predicted by logistic regressions apart from DSL internet.

                    4. All model parameters and scores have been saved successfully in a tracker.csv file.''')

        self.model = XGBClassifier()

        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }

        self.grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,                # 3-fold cross-validation
            scoring='accuracy',
            n_jobs=-1,           # Use all available cores
            verbose=1
        )

        self.grid_search.fit(self.X_train, self.y_train)
        self.y_test_pred_xgb = self.grid_search.best_estimator_.predict(self.X_test)
        # 3. Predict probabilities for the positive class (1)
        self.y_test_prob_xgb = self.grid_search.best_estimator_.predict_proba(self.X_test)[:, 1]       
        print(confusion_matrix(self.y_test, self.y_test_pred_xgb))

        #Add results to tracker
        # New row data as a list or Series
        new_row_data = ['XGB',
                        accuracy_score(self.y_test, self.y_test_pred_xgb),
                        precision_score(self.y_test, self.y_test_pred_xgb),
                        recall_score(self.y_test, self.y_test_pred_xgb),
                        f1_score(self.y_test, self.y_test_pred_xgb),
                        self.grid_search.best_params_,
                        self.grid_search.best_score_,
                        datetime.datetime.now()]

        # Add the new row
        self.tracker.loc[len(self.tracker)] = new_row_data

        #Add row to path
        self.tracker.to_csv(f"{tracker_path}",mode='a',header=False,index=False)

        #Print model metrics
        print(self.tracker.tail(1))


    def save_model(self,model_path):
        filename = f'{model_path}\\XGB_model_{datetime.datetime.now()}.joblib'.replace(":","_").replace(" ","_")
        joblib.dump(self.grid_search.best_estimator_, fr'{filename}')
