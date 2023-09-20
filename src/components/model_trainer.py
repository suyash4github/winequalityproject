import os
import sys
import pandas as pd
from src.exception import customexception
from src.logger import logging
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.utils import evaluate_models,save_object

from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config =  ModelTrainerConfig()

    def initiate_model_trainer(self,train_path,test_path):
        try:
            logging.info('Entered model trainer')
            train_df= pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            X_train = train_df.drop(['TARGET'],axis=1)
            y_train = train_df['TARGET']
            X_test = test_df.drop(['TARGET'],axis=1)
            y_test = test_df['TARGET']

            logging.info('splitted the data into x_train and x_test')

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge":Ridge(),
                "Lasso": Lasso(),
                "ElasticNet":ElasticNet()
            }
            model_report = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                          models=models)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model

            )

            predicted = best_model.predict(X_test)

            r2_sqaure = r2_score(y_test,predicted)

            return r2_sqaure

        except Exception as e:
            raise customexception(e,sys)
