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
class DataIngestionConfig:
    # Output of data ingestion: train dataset will be saved here.
    train_data_path: str=os.path.join("artifacts", "train.csv")
    
    # Output of data ingestion: test dataset will be saved here.
    test_data_path: str=os.path.join("artifacts", "test.csv")

    # Output of data ingestion: raw dataset will be saved here.
    raw_data_path: str=os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self):
        # Create an object of type: DataIngestionConfig. Gives us access to the three variables above
        self.ingestion_config=DataIngestionConfig()

    # Perform data ingestion
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion has started')

        try:
            # read the original dataset and convert it into a pandas dataframe
            df=pd.read_csv('notebooks\data\stud.csv')
            logging.info('Reading the dataset as a pandas dataframe')

            # Create the "artifacts" directory. No problem if it exists. New files will be added to this directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Store raw.csv in the location. Remove index in dataframe but keep the headers/columns
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train Test Split Initiated')
            train_set, test_set=train_test_split(df, test_size=0.2, random_state=42)

            # Move the train_set to file location as csv file
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            
            # Move the test_set to file location as csv file
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion has completed')

            # return the file path of test.csv and train.csv for transformation
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data, test_data=obj.initiate_data_ingestion()

    data_transformation_obj=DataTransformation()
    train_arr, test_arr, _=data_transformation_obj.initiate_data_transformation(train_data, test_data)

    # Perform training of models
    model_trainer=ModelTrainer()

    # Get best model name, r2_score
    print(model_trainer.initiate_model_trainer(train_data=train_data, test_data=test_data))