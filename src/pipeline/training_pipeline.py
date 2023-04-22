from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation 
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    ingestion = DataIngestion()
    train ,test = ingestion.initiate_data_ingestion()
    transformation = DataTransformation()
    train_arr,test_arr,_ = transformation.initiate_data_transforamtion(train,test)
    model = ModelTrainer()
    model.initiate_model_training(train_arr,test_arr)