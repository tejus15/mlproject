import pandas as pd
import numpy as np
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import os

class TrainPipeline:
    def __init__(self):
        pass

    def start_model_training(self, new_data):
        
        obj=DataIngestion()
        
        train_data_path=obj.ingestion_config.train_data_path
        test_data_path=obj.ingestion_config.test_data_path

        # If data has not been ingested even once, then the file path will not exist
        if not os.path.exists(train_data_path):
            train_data_path, test_data_path=obj.initiate_data_ingestion()
        
        # Append to CSV
        new_data.to_csv(train_data_path, index=False, mode='a', header=False)

        data_transformation_obj=DataTransformation()
        train_arr, test_arr, _=data_transformation_obj.initiate_data_transformation(train_data_path, test_data_path)
        
        # Perform training of models
        model_trainer=ModelTrainer()

        # Get best model name, r2_score
        print(model_trainer.initiate_model_trainer(train_array=train_arr, test_array=test_arr))