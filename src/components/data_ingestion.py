import sys
import os

from src.logger import logging
from src.exception import ZomatoException
from src.config.configuration import *

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH
    raw_data_path:str = RAW_FILE_PATH
    
class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig
        
    def initiate_data_ingestion(self):
        
        logging.info("="*50)
        logging.info("Initiate Data Ingestion config")
        logging.info("="*50)
        
        try:
            df = pd.read_csv(DATA_FILE_PATH,parse_dates = ["Order_Date"])
            logging.info("Dataset in readed")
            
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            

            logging.info(f"Raw data file path: {[self.data_ingestion_config.raw_data_path]}")
            
            logging.info("Train test split started")
            train_set,test_set = train_test_split(df,test_size=0.30,random_state=32)
            
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            
            logging.info("Ingestion of Data completed")
            
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )            
            
        except Exception as e:
            logging.info("Exception occured in Data Ingestion")
            raise ZomatoException(e,sys)
        
    
if __name__ == "__main__":
    ingestion = DataIngestion()
    ingestion.initiate_data_ingestion()
