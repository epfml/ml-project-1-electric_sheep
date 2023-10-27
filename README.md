# Project 1: CVD prediction 
## Team
Authors: Ismail Sahbane, Yanruiqi Yang, Ameer Elkhayat
## Project Structure
### Data Preprocessing 
Filtering data, Feature generation, Cleaning missing data, etc.
### Training and Optimization
Optimize the weights to minimize the loss function.
### Prediction
Transform the output of the model to prediction labels.
### Local Validation + Testing
Split the data to the training/test dataset and evaluate the performance of the model.

## Files:
### implementations.py
Functions of Implemented basic algorithms: mean_squared_error_gd, mean_squared_error_sgd, least_squares ridge_regression, logistic_regression, reg_logistic_regression.
### run.py
Execution Program of the system
### test.py
Testing file
### utils.py
### ml_methods.py

## Data Processing 
### Scalar Data:
fill_na (with mean), normalization, build the polynomial predictors
### Categorical Data:
One-hot-encoding

## Training Model
Logistic regression with ADAM-optimizer SGD

## Results
Our highest F-score is 0.445, with submission ID #240035 on Aicrowd.

