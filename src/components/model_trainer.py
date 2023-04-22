import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from src.config.configuration import MODEL_FILE_PATH
from src.exception import ZomatoException
from src.logger import logging
from src.utils import *

from src.components import data_ingestion
from src.components import data_transformation

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info("="*50)
            logging.info("Model Training Intiated")
            logging.info("="*50)
            
            X_train,y_train,X_test,y_test = (train_array[:,:-1],
                                             train_array[:,-1],
                                             test_array[:,:-1],
                                             test_array[:,-1]
                                             )
            
            models = {
                'LinearRegression' : LinearRegression(),
                'Ridge' : Ridge(),
                'Lasso' : Lasso(),
                'ElasticNet' : ElasticNet()
            }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            logging.info("="*50)
            logging.info("Model Report")
            logging.info("="*50)    
            logging.info(model_report)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            logging.info("="*50)
            logging.info(f"Best Model: {best_model_name}, R2 score: {best_model_score}")
            logging.info("="*50)
            
            save_obeject(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model)
            
        except Exception as e:
            logging.info("Exception occured in initiate model training")
            raise ZomatoException(e,sys)

