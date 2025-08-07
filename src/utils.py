import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException
from sklearn.metrics import r2_score

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
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        reports={}
        # loop through the models dictionary
        for i in range(len(list(models))):
            
            # Get a model
            model=list(models.values())[i]

            # Train the model
            model.fit(X_train, y_train)
            
            # Predict the output using test dataset
            y_test_pred=model.predict(X_test)

            # get r2 score on y_test predictions
            test_model_score=r2_score(y_test, y_test_pred)

            # Add entry to report dictionary
            reports[list(models.keys())[i]]=test_model_score
        
        return reports
    
    except Exception as e:
        raise CustomException(e, sys)