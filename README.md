# Project 1: CVD prediction 
## Team:
Authors: Ismail Sahbane(Computer Science), Yanruiqi Yang(Management of Technology), Ameer Elkhayat(Neuro-X)


## Project Structure:
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
Functions of implemented basic algorithms: 
* mean_squared_error_gd 
* mean_squared_error_sgd
* least_squares ridge_regression
* logistic_regression
* reg_logistic_regression.
### run.py
Execution Program of the system
### test.py
Testing file
### utils.py

### ml_methods.py
* reg_logistic_regression_sgd: logistic regression with general stochastic gradient descend method
* reg_logistic_regression_adam: logistic regression with SGD method and Adam as the optimizer
* compute_MSE_gradient_and_loss: computer the gradient and loss using Minimum Squared Error(MSE) as loss
* compute_cross_entropy_gradient_and_loss: computer the gradient and loss using logistic loss
* sigmoid: sigmoid function 
* logistic_predict: prediction function, map the output probability to a prediction label
* batch: function to generate batches for SGD

## Data Processing: 
### Scalar Data:
fill_na (with mean), normalization, build the polynomial predictors
### Categorical Data:
One-Hot-Encoding

## Training Model:
Logistic regression with ADAM-optimizer SGD

## Results:
Our highest F-score is 0.445, with submission ID #240035 on Aicrowd.

