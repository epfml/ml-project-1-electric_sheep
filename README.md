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

### run.py
Execution Program of the system

### test.py
Testing file

### implementations.py
Functions of implemented basic algorithms: 
* **mean_squared_error_gd**
* **mean_squared_error_sgd**
* **least_squares ridge_regression**
* **logistic_regression**
* **reg_logistic_regression**

### utils.py
Input and output:
* **load_data**: loading data from the given dataset
* **create_csv_submission**: saving the output to file

Data processing: 
* **normalize**: normalize the data of a feature
* **rows_with_all_features**: find the rows with all features non-nan
* **remove_rows_with_missing_features**: remove rows with at least one missing feature in x and y
* **remove_rows_with_too_many_missing_features**: removes the rows for which the fraction of nan features is bigger than f 
* **replace_missing_features_with_mean**: replace all nan values in x with the mean of their column
* **build_poly**: construct the polynomial predictor for a feature
* **one_hot_encoding**: generate one-hot-encoding features for a categorical feature
  
Testing and Validation:
* **split_data**: split the dataset to training/test data, given a split ratio
* **evaluate**: compute and display the validation results, with the metric F1-score
* **find_optimal_c**: find the optimal c to predict the labels
* **plot_f1_to_c**: plot the curve of F1-score in different cutoff probability cases
* **generate_linear_data_with_gaussian_noise**: small util used to test functions with random data
  
### ml_methods.py
* **reg_logistic_regression_sgd**: logistic regression with general stochastic gradient descend method
* **reg_logistic_regression_adam**: logistic regression with SGD method and Adam as the optimizer
* **compute_MSE_gradient_and_loss**: computer the gradient and loss using Minimum Squared Error(MSE) as loss
* **compute_cross_entropy_gradient_and_loss**: computer the gradient and loss using logistic loss
* **sigmoid**: sigmoid function 
* **logistic_predict**: prediction function, map the output probability to a prediction label
* **batch**: function to generate batches for SGD

## Data Processing: 
### Scalar Data:
fill_na (with mean), normalization, build the polynomial predictors
### Categorical Data:
One-Hot-Encoding

## Training Model:
Logistic regression with ADAM-optimizer SGD

## Results:
Our highest F-score is 0.445, with submission ID #240035 on Aicrowd.

