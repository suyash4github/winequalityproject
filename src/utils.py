from src.exception import customexception
from src.logger import logging

import os
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_predict = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise customexception(e,sys)
    
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise customexception(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise customexception(e, sys)