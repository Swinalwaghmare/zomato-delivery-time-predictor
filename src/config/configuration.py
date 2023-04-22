from src.constant import *
import os

ROOT_DIR = ROOT_DIR_KEY

DATA_FILE_PATH = os.path.join(ROOT_DIR,DATA_DIR,DATA_DIR_KEY)

# Data ingestion config
TRAIN_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,
                               TRAIN_DATA_DIR_KEY)

TEST_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,
                              TEST_DATA_DIR_KEY)

RAW_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,DATA_INGESTION_KEY,
                             RAW_DATA_DIR_KEY)

PREPROCESSING_OBJ_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                                      DATA_TRANSFORMATION_ARTIFACT,
                                      DATA_TRANSFORMATION_PREPROCESSING_OBJ)

MODEL_FILE_PATH = os.path.join(ROOT_DIR,ARTIFACT_DIR_KEY,
                               MODEL_ARTIFACT,MODEL_OBJECT)