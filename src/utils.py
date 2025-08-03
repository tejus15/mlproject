import os
import sys
import numpy as np
import pandas as pd
import dill
from src.exception import CustomException

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