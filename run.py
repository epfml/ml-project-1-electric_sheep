import utils
import ml_methods

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)



#=================================HYPER PARAMETERS=================================#
N_train = 10000
iters = 400
lambda_ = 1e-7
learning_rate = 0.001
batch_size = 512



#========================FEATURE SELECTION AND DATA LOADING========================#
x_features = [0, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 88, 89, 96, 97, 98, 104, 105, 108, 109, 110, 117, 118, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 145, 147, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 197, 199, 200, 201, 202, 203, 204, 205, 206, 215, 216, 249, 252, 254, 267, 268, 269, 270, 271, 272, 277, 278, 292, 293, 296, 297, 303, 304, 305]
d = len(x_features) - 1
c = np.full(d, True)
c[-18:] = False # All the scalar features



# load the data
x_train, x_test, y_train, train_ids, test_ids = utils.load_data(
    x_train_path="dataset/x_train.csv", 
    y_train_path="dataset/y_train.csv", 
    x_test_path="dataset/x_test.csv", 
    max_rows_train=N_train, 
    max_rows_test=10, 
    x_features=x_features
)



#=======================PERFORM ALL THE DATA PROCESSING=======================#

#in order to have the same category set for all categorical features in training and testing, we need to combine them before processing
x = np.concatenate((x_train, x_test), axis=0)

x_scalar = x[:, ~c]
x_scalar = utils.replace_missing_features_with_mean(x_scalar)
tx_scalar = utils.build_poly(x_scalar, 6)
tx_scalar[:, 1:] = utils.normalize(tx_scalar[:, 1:]) # we shouldn't normalize the first column (1 vector)

x_categorical = x[:, c]
x_categorical = utils.one_hot_encoding(x_categorical)

tx = np.concatenate((tx_scalar, x_categorical), axis=1)

tx_train = tx[:N_train]
tx_test = tx[N_train:]



#======================DIVIDE THE DATASET FOR VALIDATION======================#
ratio = 0.8
tx_train_train, tx_train_test, y_train_train, y_train_test = utils.split_data(ratio, tx_train, y_train)

N_train_train = tx_train_train.shape[0]
d = tx_train_train.shape[1]


#=================TRAINING USING LOGISTIC REGRESSION WITH ADAM=================#
initial_w = np.zeros(d)
loss, w = ml_methods.reg_logistic_regression_adam(y_train_train, tx_train_train, lambda_, initial_w, iters, learning_rate, batch_size)



#=================FIND THE BEST CUTOFF USING THE VALIDATION SET=================#
optimal_c, optimal_f1 = utils.find_optimal_c(tx_train_test, y_train_test, w)



#====================PREDICT THE LABELS OF THE UNLABELED DATA====================#
test_predictions = ml_methods.logistic_predict(tx_test, w, c=optimal_c) # should be y values between 0 and 1
test_predictions = test_predictions * 2 - 1 # should now be between -1 and 1 as desired

#write to csv
utils.create_csv_submission(test_ids, test_predictions, f"submission.csv")



#to delete
utils.plot_f1_to_c(tx_train_test, w, y_train_test, delta=0.001)
print(f"optimal f1 = {optimal_f1}")
