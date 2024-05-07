import os
import sys
from source.exception import CustomException
from source.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from source.components.data_transformation import DataTrasformation
from source.components.model_trainer import ModelTrainer,ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('entered data ingestion method or component')
        try:
            df=pd.read_csv(r'notebook\data\stud.csv')
            logging.info('read dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            train_set,test_set=train_test_split(df,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info('data ingestion is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTrasformation()
    train_arr,test_arr,_=data_transformation.initiating_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))