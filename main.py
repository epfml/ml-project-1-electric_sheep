import utils
import implementations

import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)

"""
    THINGS WE CAN DO: (TODO)
    - use adam optimizer or something else that inteligently changes the learning rate during the training process
    - understand why it doesn't work with scalar features *
    - understand why it doesn't work with sgd, and find ways to train quicker *
    - modularize things like : process features, split dataset, test accuracy
    - automatically find optimal lambda, learning rates, etc. 
    - function to write to csv
    - rewrite least_squares using np.linalg.solve (and maybe also ridge_regression)
    - devise optimal feature choice and processing (consider removing features with very small weight associated)

    and maybe:
    - finish poly feature expansion (only if we can make scalar values work)
    - extrapolate for the data we don't have, stuff like that
    - remove outliers 
    - have some command to interrupt training and stop
"""

# feature selection

# with scalar features
#usecols = [26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 87, 88, 95, 96, 97, 103, 104, 107, 108, 109, 116, 117, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 200, 201, 202, 203, 204, 205, 214, 215, 248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304] # False : 248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304 (last 18 elems)
#c = np.full(len(usecols), True)
#c[np.arange(118, 136)] = False # All the scalar features

#without them
#usecols = [26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 87, 88, 95, 96, 97, 103, 104, 107, 108, 109, 116, 117, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 200, 201, 202, 203, 204, 205, 214, 215]
x_features = [0, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 88, 89, 96, 97, 98, 104, 105, 108, 109, 110, 117, 118, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 145, 147, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 197, 199, 200, 201, 202, 203, 204, 205, 206, 215, 216]
c = np.full(len(x_features) - 1, True)

#with only them
#usecols = [248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304]
#c = np.full(len(usecols), False)

# load the data
x_train, x_test, y_train, train_ids, test_ids = utils.load_data(
    x_train_path="dataset/x_train.csv", 
    y_train_path="dataset/y_train.csv", 
    x_test_path="dataset/x_test.csv", 
    max_rows_train=10000, 
    max_rows_test=None, 
    x_features=x_features
)

# Data Processing #NOTE idea: if there's a lot of data processing, we could do it once and save it to a new .csv file, and use this as dataset to train faster
#x, y = utils.remove_rows_with_missing_features(x, y)
def all_x_processing(x):
    print(f"x shape = {x.shape}, c shape = {c.shape}")
    x = utils.replace_missing_features_with_mean(x, c) # we replace with mean and normalize only for the scalar features
    x = utils.normalize(x, c)
    x = utils.one_hot_encoding(x, c)
    #tx = x
    #tx = utils.build_poly(x, 1)
    N = x.shape[0] 
    tx = np.concatenate((np.ones((N, 1)), x), axis=1)

    return tx


# problem : tx_train and tx_test might not contain samples from all possible categories for all features. therefore, one_hot_encoding may return inconsistent results.
# solution : either we concatenate x_train and x_test and do all the processing on it. that would be very sensible, and make the mean-replace and normalisation more robust.

N_train = x_train.shape[0]
x = np.concatenate((x_train, x_test), axis=0)
tx = all_x_processing(x)
tx_train = tx[:N_train]
tx_test = tx[N_train:]


#divide dataset
ratio = 0.8
split = int(np.floor(ratio * N_train))
tx_train_test = tx_train[split:]
y_train_test = y_train[split:]
tx_train_train = tx_train[:split]
y_train_train = y_train[:split]

N_train_train = tx_train_train.shape[0]
d = tx_train_train.shape[1]


#train by calling the right function
initial_w = np.zeros(d)
loss, w = implementations.reg_logistic_regression(y_train_train, tx_train_train, 0, initial_w, 5000, 0.4)

#print(f"Loss -> {loss}")
#print(f"final w : {w}")

#training accuracy
train_train_predictions = implementations.logistic_predict(tx_train_train, w)
print(f"predictions = {train_train_predictions}")
print(f"and y was : {y_train_train}")
train_train_errors = np.count_nonzero(train_train_predictions - y_train_train)
print(f"{train_train_errors} errors for {N_train_train} samples")
train_train_accuracy = 1. - train_train_errors / N_train_train
print(f"Accuracy : {100. * train_train_accuracy}%")
print(f"y 1 count was {y_train_train.sum()}, so predicting only 0 yields {100. * (1. - y_train_train.sum() / N_train_train)}% accuracy")




#testing accuracy
N_train_test = y_train_test.shape[0]
train_test_predictions = implementations.logistic_predict(tx_train_test, w)
train_test_errors = np.count_nonzero(train_test_predictions - y_train_test)
train_test_accuracy = 1. - train_test_errors / N_train_test
print(f"\n\n=============================== TESTING ===============================")
print(f"{train_test_errors} errors for {N_train_test} samples")
print(f"Accuracy : {100. * train_test_accuracy}%")
print(f"y 1 count was {y_train_test.sum()}, so predicting only 0 yields {100. * (1. - y_train_test.sum() / N_train_test)}% accuracy")


#visualize?


# generate submission testing data
print(f"tx_test.shape={tx_test.shape}, w.shape={w.shape}")
test_predictions = implementations.logistic_predict(tx_test, w) # should be y values between 0 and 1
test_predictions = test_predictions * 2 - 1 # should now be between -1 and 1 as desired

#write to csv
utils.create_csv_submission(test_ids, test_predictions, "submission.csv")