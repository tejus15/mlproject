import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline
from src.utils import CustomData, load_object
from src.logger import logging
from src.exception import CustomException

# Gives us entry point
application=Flask(__name__)

app=application

# For better user experience (faster response), run training in background:
# Create a thread pool executor globally
executor = ThreadPoolExecutor(max_workers=2)

def retrain_model_async(pred_df, prediction_result):
    """Background function for model retraining"""
    try:
        new_data = pred_df.copy()
        new_data['math_score'] = prediction_result
        
        train_pipeline = TrainPipeline()
        train_pipeline.start_model_training(new_data=new_data)
        
        logging.info('Model trained using user input (background)')
        
    except Exception as e:
        raise CustomException(f'Error in background model retraining: {e}')

# Route to a home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # If we have a GET request, I will default to home.html (index.html is the home page by default)
    if request.method=='GET':
        # home.html will contain simple form fields for user iput
        return render_template('home.html')
    # POST method
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )
        
        # Convert the user input into a data frame
        df=data.get_data_as_data_frame()
        print(df)

        predict_pipeline=PredictPipeline()

        # Get predictions
        results=predict_pipeline.predict(df)
        
        # Submit retraining task to background thread
        executor.submit(retrain_model_async, df, results[0])

        return render_template('home.html', results=results[0])
    
if __name__=="__main__":
    app.run(host="0.0.0.0")