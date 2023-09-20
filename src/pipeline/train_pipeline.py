from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig










if __name__=="__main__":
    obj=DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_path,test_path))