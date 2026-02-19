import pandas as pd
import numpy as np


class PreProcessTrainingData():
    def __init__(self,path):
        self.data = pd.read_csv(f"{path}",index_col=None)
        print(self.data.head())
        self.data.info()
        print(self.data.describe())

    def check_nulls(self):
        print("Detected Nulls:")
        print(self.data[self.data.isnull().any(axis=1)])

    def check_duplicates(self):
        print("Detected Duplicates:")
        print(self.data[self.data.duplicated()])

    def pre_process_categorical_numeric_columns(self,drop_columns,categorical_columns,numeric_columns,target_column,save_data_path):
        self.data_fine_tune = self.data.drop(drop_columns, axis = 1)
        self.data_fine_tune[target_column]=self.data_fine_tune[target_column].map({"Yes":1,"No":0})
        self.data_fine_tune_encoded = pd.get_dummies(self.data_fine_tune, columns=categorical_columns,drop_first=False,dtype='int')
        self.data_fine_tune_encoded.columns = self.data_fine_tune_encoded.columns.str.replace(' ', '_')
        self.data_fine_tune_encoded.info()
        self.data_fine_tune_encoded_numeric = self.data_fine_tune_encoded[numeric_columns].apply(pd.to_numeric,errors='coerce')
        for k,v in self.data_fine_tune_encoded_numeric.isnull().sum().items():
            if int(v)!=0:
                Nulls=self.data_fine_tune_encoded_numeric[np.isnan(self.data_fine_tune_encoded_numeric[k])]
                print(Nulls)
                self.data_fine_tune_encoded.drop(labels=Nulls.index, axis=0, inplace=True)
                self.data_fine_tune_encoded[[k]] = self.data_fine_tune_encoded[[k]].apply(pd.to_numeric, errors='coerce')
        print(self.data_fine_tune_encoded.head())
        self.data_fine_tune_encoded.to_csv(save_data_path,index=False)

