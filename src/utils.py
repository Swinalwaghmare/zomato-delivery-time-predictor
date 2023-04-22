import os
import sys
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from src.exception import ZomatoException
from src.logger import logging

def save_obeject(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        logging.info('Exception occured while saving an object')
        raise ZomatoException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        
        for i in range(len(models)):
            model = list(models.values())[i]
            
            model.fit(X_train,y_train)
            
            y_test_pred = model.predict(X_test)
            
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
        
        return report
            
    except Exception as e:
        logging.info("Exception occure while evaluation of model")
        raise ZomatoException(e,sys)