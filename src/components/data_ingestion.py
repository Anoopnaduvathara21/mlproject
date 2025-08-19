import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class  DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv") #creates a simple class for configs.
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"data.csv")

class DataIngestion:                                     #Initializes with the config.
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):                  
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')   #Reads the raw dataset.
            logging.info('Read the dataset as data frame')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)   #Ensures artifacts/ folder exists.

            df.to_csv(self.ingestion_config.raw_data_path,index = False,header=True)       #Saves the raw dataset.

            logging.info("Train test split is initiated")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)       #Splits into 80% train, 20% test.

            train_set.to_csv(self.ingestion_config.train_data_path,index = False, header =True)  #Saves train/test CSV files.

            test_set.to_csv(self.ingestion_config.test_data_path,index = False, header =True)

            logging.info("ingestion of the data is completed")

            return(
                    self.ingestion_config.train_data_path,       #Returns the two paths.
                    self.ingestion_config.test_data_path
                )
        except Exception as e:
                raise CustomException(e,sys)


if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()                   #Runs ingestion → transformation → training in sequence.

    data_transformation = DataTransformation()
    train_arr, test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))