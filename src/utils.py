import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, KFold, StratifiedKFold

def save_object(file_path, obj):
    try:
        # Get directory component of file path
        dir_path=os.path.dirname(file_path)

        # Create directory
        os.makedirs(dir_path, exist_ok=True)

        # Open the pickle file in wb mode
        with open(file_path, "wb") as file_obj:
            # Pickle an object to a file. 
            dill.dump(obj, file_obj)

    except CustomException as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        reports={}
        best_models={}
        # loop through the models dictionary
        for i in range(len(list(models))):
            
            # Get a model
            model=list(models.values())[i]

            # Get model specific hyperparameters
            param=params[list(models.keys())[i]]

            # Create randomized search cv object to initiate hyperparameter tuning
            random_cv=RandomizedSearchCV(model, param, cv=5)

            # Train the model
            random_cv.fit(X_train, y_train)
            
            # Get the best model of each algorithm by setting the best parameters
            model.set_params(**random_cv.best_params_)
            
            logging.info(f"Best parameters for {list(models.keys())[i]} : {dict(random_cv.best_params_)}")

            # Train the tuned model using training set
            model.fit(X_train, y_train)

            # Add it to the dictionary of best models
            best_models[list(models.keys())[i]]=model

        # Use KFold for regression
        cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in best_models.items():
            logging.info(f"Evaluating {name} with cross-validation...")
        
            r2_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='r2')

            # Get mean r2 score for each model. used for comparison. 
            reports[name] = r2_scores.mean()
            logging.info(f"Mean r2 score for {name} : {reports[name]}")
            
        return reports, best_models
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with(open(file_path, "rb")) as file_obj:
            return dill.load(file_obj)            
    except Exception as e:
        raise CustomException(e, sys)
    
# Maps input in HTML Form to backend
class CustomData:
    def __init__(
            self, 
            gender: str,
            race_ethnicity: str,
            parental_level_of_education,
            lunch: str,
            test_preparation_course: str,
            reading_score: int,
            writing_score: int
        ):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict={
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)