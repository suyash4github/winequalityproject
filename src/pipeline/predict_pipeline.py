import os
import sys
import pandas as pd
from src.exception import customexception
from src.logger import logging

from src.utils import load_object


class Predictpipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path ='artifacts\model.pkl'
            model = load_object(file_path=model_path)
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise customexception(e,sys)

class CustomData:
    def __init__(self,
                 fixed_acidity: float, 
                 volatile_acidity: float, 
                 citric_acid:float,
                 residual_sugar:float,
                 chlorides:float,
                 free_sulfur_dioxide: float,
                 total_sulfur_dioxide: float,
                 density: float,
                 pH: float,
                 sulphates: float,
                 alcohol: float):
    
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
    
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict ={
                "fixed_acidity" : [self.fixed_acidity],
                "volatile_acidity":[self.volatile_acidity],
                "citric_acid" : [self.citric_acid],
                "residual_sugar":[self.residual_sugar],
                "chlorides":[self.chlorides],
                "free_sulfur_dioxide":[self.free_sulfur_dioxide],
                "total_sulfur_dioxide":[self.total_sulfur_dioxide],
                "density":[self.density],
                "pH":[self.pH],
                "sulphates":[self.sulphates],
                "alcohol":[self.alcohol]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise customexception(e,sys)
    


        
        




