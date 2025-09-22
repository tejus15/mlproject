# Student Math Performance Prediction Project
## Overview
This project aims to predict how well students will perform on math tests using various demographic, educational, and academic factors. By analyzing patterns in student data, we can identify key factors that influence math performance and build a predictive model to help educators and students better understand potential outcomes.

## Project Goals
The main objective is to create a machine learning model that can accurately predict a student's math test score based on:

<i>Demographic information (gender, race, ethnicity) <br/>
Educational background (parental education, test prep completion, school lunch type)<br/>
Academic performance (reading and writing test scores)<br/></i>

## Dataset Description
Dataset can be found here: https://www.kaggle.com/datasets/spscientist/students-performance-in-exams/data<br/>
Features Used for Prediction:
### Demographic Data:

Gender<br/>
Race/Ethnicity
### Educational Background:

Parental level of education<br/>
Test preparation course completion status<br/>
Lunch type (standard vs. free/reduced)<br/>
<i>Note about lunch programs: Students from low-income families are eligible for meal assistance programs. Free meals are provided to children in households with incomes below 130% of the poverty level or those receiving SNAP/TANF benefits. Reduced-price meals are available for families with incomes between 130-185% of the poverty line.</i>

### Academic Performance:

Reading test scores (0-100) <br/>
Writing test scores (0-100)
### Target Variable:

Math test scores (0-100)

## Technical Implementation
### Data Processing Pipeline
#### 1. Data Preparation

Split the dataset into training and testing sets<br/>
Created separate files for train and test datasets to maintain data integrity<br/>
#### 2. Preprocessing Pipeline 
The data preprocessing was handled differently for numeric and categorical variables:

##### Numeric Variables (Reading/Writing Scores):
Simple imputation using median strategy to handle missing values<br/>
Standard scaling for normalization<br/>
##### Categorical Variables (Demographics, Education Background):
Simple imputation using most frequent value strategy<br/>
One-hot encoding (without dropping categories)<br/>
Standard scaling for consistency<br/>

#### 3. Pipeline Integration

Used ColumnTransformer to combine both preprocessing pipelines<br/>
Applied the complete preprocessor to the training dataset<br/>
### Model Selection and Training
Algorithms Tested:

Ridge Regression <br/>
Decision Tree Regressor <br/>
XGBoost Regressor <br/>
Gradient Boosting Regressor <br/>
CatBoost Regressor <br/>
AdaBoost Regressor <br/>
Random Forest Regressor <br/>
K-Nearest Neighbors (KNN) Regressor <br/>
### Model Optimization Process:

#### Hyperparameter Tuning: 
Applied RandomizedSearchCV to find optimal parameters for each algorithm
#### Cross-Validation: 
Performed 5-fold cross-validation to ensure model reliability
#### Evaluation Metric: 
Used R² score to assess model performance
#### Model Selection: 
Ridge Regression achieved the highest R² score and was selected as the final model
#### Model Persistence: 
Saved the trained model as a pickle file for deployment

### Web Application Development
#### Flask Web App Features:

User-friendly form interface for input collection<br/>
Real-time prediction capability<br/>
Integration with the trained model (pickle file)<br/>
Clean, responsive design for easy interaction<br/>
#### Input Form Fields:

Student demographic information<br/>
Educational background details<br/>
Reading and writing test scores<br/>
Instant math score prediction output<br/>
### Deployment Strategy
#### Containerization & Cloud Deployment:

<b>Docker Image Creation:</b> Built a Docker container for the Flask application <br/>
<b>Private Repository:</b> Stored the Docker image in Amazon ECR (Elastic Container Registry)<br/>
<b>Cloud Deployment:</b> Deployed the containerized application to an EC2 instance <br/>
<b>Scalability:</b> Cloud-based deployment ensures reliable access and scalability <br/>

## Key Results
### Best Performing Model: 
Ridge Regression
### Evaluation Metric: 
R² Score=0.87
### Model Performance: 
Achieved optimal cross-validation scores through systematic hyperparameter tuning
### Deployment Status: 
Successfully deployed as a cloud-based web application

## Usage
The deployed web application allows users to:

Enter student information through an intuitive web form <br/>
Receive instant math performance predictions <br/>
Understand the impact of various factors on academic outcomes <br/>

## Future Enhancements
Integration of additional educational factors <br/>
Implementation of model interpretation features<br/>
Advanced visualization of prediction factors<br/>
Real-time model retraining capabilities<br/>
This project demonstrates a complete end-to-end machine learning workflow, from data preprocessing to cloud deployment, providing valuable insights into factors affecting student academic performance in mathematics.

## Screenshot of the Web Application
### Website On Laptop:
<img width="1905" height="1025" alt="image" src="https://github.com/user-attachments/assets/fdb9a4ba-bf28-4d89-a59e-3c9d6e3e2091" />

### Website Prediction Form:
<img width="1890" height="977" alt="image" src="https://github.com/user-attachments/assets/3014dbd1-e402-4c77-857c-0b51b8025b8d" />

### Website Score Prediction Based on User Input:
<img width="1903" height="1017" alt="image" src="https://github.com/user-attachments/assets/e8e36df9-20e8-477d-b8da-b84c46e2bf3f" />

### Website on ipad:
![60843](https://github.com/user-attachments/assets/00825661-d8f9-4fbb-a4f7-b6db718dfcf1)
