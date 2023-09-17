import os
import sys
from src.exception import customexception
from src.logger import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion')
        try:
            df = pd.read_csv(os.path.join('data','winequality.csv'))
            logging.info('Read the dataset')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train test split initiated')
            train_set,test_set=train_test_split(df,random_state=42,test_size=0.2)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info('Data ingestion completed.')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise customexception(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()
