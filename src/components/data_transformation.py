import os 
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline

from src.exception import ZomatoException
from src.logger import logging
from src.components import data_ingestion
from src.config.configuration import PREPROCESSING_OBJ_PATH
from src.utils import save_obeject

@dataclass 
class DataTransformationConfig:
    preprocessor_obj_file_path = PREPROCESSING_OBJ_PATH
    
class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info("="*50)
            logging.info("Data transformation object creation started")
            logging.info("="*50)
            
            categorical_columns = ['Weather_conditions' ,'Road_traffic_density']
            numerical_columns = ['Delivery_person_Age','Delivery_person_Ratings','Vehicle_condition',
                                 'multiple_deliveries','Dist_from_Rest_to_deli_loc']
            
            # Define the custome ranking for each ordinal variable
            Weather_conditions_categories = ['Sunny','Cloudy','Fog','Windy','Sandstorms','Stormy']
            Road_traffic_density_categories = ['Low','Medium','High','Jam']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions_categories,
                                                                 Road_traffic_density_categories])),
                    ('scaler',StandardScaler())
                ]
            )
            
            preprocessor = ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            
            return preprocessor
            
        except Exception as e:
            logging.info('Exception occured in get data transformation object')
            raise ZomatoException(e,sys)
    
    '''
    def extracting_order_date(self,df, feature):
       df[feature]=pd.to_datetime(df[feature],format="%d-%m-%Y")
       df['day'] = df[feature].dt.day
       df['month'] = df[feature].dt.month
       df['days_of_week'] = df[feature].dt.day_name()
       df.drop([feature],axis=1,inplace=True)
    
    #def extr_time_hr_min(self,df,feature):
        df[feature] = df[feature].str.replace(".",":")
        df['Hours'] = df[feature].str.split(':').str[0]
        df['Min'] = df[feature].str.split(':').str[1].str[:2]
        df[feature] = df['Hours'] + ":" + df['Min']
        df.drop(['Hours','Min'],axis=1,inplace=True)
        df[feature] = pd.to_datetime(df[feature],format='%H:%M',errors='coerce')
        invalid_rows = df.loc[df[feature].isna()]
        df.loc[invalid_rows.index, feature] = invalid_rows[feature].apply(lambda x: x.replace(minute=0))
        df[f"{feature}_hour"] = df[feature].dt.hour
        df[f'{feature}_min'] = df[feature].dt.minute
        df.drop([feature],axis=1,inplace=True)
    '''
    def distance_numpy(self,df,lat1, lon1, lat2, lon2):
        p = np.pi/180
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2]-df[lon1])*p))/2
        df['Dist_from_Rest_to_deli_loc'] = 12742 * np.arcsin(np.sqrt(a))
        
    def initiate_data_transforamtion(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path) 
            test_df = pd.read_csv(test_path)
            
            '''
            logging.info("Dropping ID column")
            train_df.drop(['ID'],axis=1,inplace=True)
            test_df.drop(['ID'],axis=1,inplace=True)
            
            logging.info("creating Delivery person ID")
            train_df["City_name"] = train_df['Delivery_person_ID'].str.split('RE').str[0]
            train_df.drop(['Delivery_person_ID'],axis=1,inplace=True)
            test_df["City_name"] = test_df['Delivery_person_ID'].str.split('RE').str[0]
            test_df.drop(['Delivery_person_ID'],axis=1,inplace=True)
            
            logging.info("Dropping invalid latitude and longitude column")
            Rest_lat_index = train_df[train_df['Restaurant_latitude'] <= 0].index.to_list()
            train_df.drop(index=Rest_lat_index,inplace=True)
            Rest_lat_index = test_df[test_df['Restaurant_latitude'] <= 0].index.to_list()
            test_df.drop(index=Rest_lat_index,inplace=True)
            
            logging.info("Creating feature on latitude nad longitude")
            self.distance_numpy(train_df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')
            self.distance_numpy(test_df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')

            logging.info("Extracting day, month and day of week")
            self.extracting_order_date(train_df,'Order_Date')
            self.extracting_order_date(test_df,'Order_Date')
            
            logging.info("Dropping NaN from Time orderd column")
            train_df.dropna(subset=['Time_Orderd'],inplace=True)
            test_df.dropna(subset=['Time_Orderd'],inplace=True)
            
            logging.info("Creating feature from time_orderd hour and min")
            self.extr_time_hr_min(train_df,'Time_Orderd')
            self.extr_time_hr_min(test_df,'Time_Orderd')
            
            logging.info("Creating feature from Time_Order_picked hour and min")
            self.extr_time_hr_min(train_df,'Time_Order_picked')
            self.extr_time_hr_min(test_df,'Time_Order_picked')
            '''
            
            logging.info("Creating feature on latitude nad longitude")
            self.distance_numpy(train_df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')
            self.distance_numpy(test_df,'Restaurant_latitude','Restaurant_longitude', 
                                'Delivery_location_latitude','Delivery_location_longitude')

            
            columns_to_drop = ['ID','Delivery_person_ID','Restaurant_latitude','Restaurant_longitude',
                               'Delivery_location_latitude','Delivery_location_longitude','Order_Date',
                               'Time_Orderd','Time_Order_picked','Type_of_order','Type_of_vehicle','Festival','City']
            
            train_df.drop(columns=columns_to_drop,axis=1,inplace=True)
            test_df.drop(columns=columns_to_drop,axis=1,inplace=True)
            
            logging.info("train test data reading completed")
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Train Dataframe Head: \n{test_df.head().to_string()}')
            
            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object()
            
            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name]
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            # Transforming using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info("transformation completed")
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)] 
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_obeject(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj)
            
            logging.info("Preprocessor file saved")
            
            return(train_arr,
                   test_arr,
                   self.data_transformation_config.preprocessor_obj_file_path)
            
        except Exception as e:
            logging.info("Exception occured in initiate data transformation")
            raise  ZomatoException(e,sys)
            
if __name__ == "__main__":
    ingestion = data_ingestion.DataIngestion()
    train ,test = ingestion.initiate_data_ingestion()
    transformation = DataTransformation()
    transformation.initiate_data_transforamtion(train,test)