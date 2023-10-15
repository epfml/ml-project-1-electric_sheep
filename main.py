import utils
import implementations

import numpy as np
import sys
#np.set_printoptions(threshold=sys.maxsize)

"""
    THINGS WE CAN DO: (TODO)
    - use adam optimizer
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
"""

# feature selection

# with scalar features
#usecols = [26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 87, 88, 95, 96, 97, 103, 104, 107, 108, 109, 116, 117, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 200, 201, 202, 203, 204, 205, 214, 215, 248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304] # False : 248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304 (last 18 elems)
#c = np.array([True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,])
#c = np.full(len(usecols), True)
#c[np.arange(118, 136)] = False # All the scalar features

#without them
usecols = [26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 87, 88, 95, 96, 97, 103, 104, 107, 108, 109, 116, 117, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 200, 201, 202, 203, 204, 205, 214, 215]
c = np.full(len(usecols), True)

#with only them
#usecols = [248, 251, 253, 266, 267, 268, 269, 270, 271, 276, 277, 291, 292, 295, 296, 302, 303, 304]
#c = np.full(len(usecols), False)

# load the data
x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv", max_rows=6250, usecols=usecols)
# Data Processing #NOTE idea: if there's a lot of data processing, we could do it once and save it to a new .csv file, and use this as dataset to train faster
#x, y = utils.remove_rows_with_missing_features(x, y)
x = utils.replace_missing_features_with_mean(x, c) # we replace with mean and normalize only for the scalar features
x = utils.normalize(x, c)
x = utils.one_hot_encoding(x, c)
#tx = x
#tx = utils.build_poly(x, 1)
N = x.shape[0] 
tx = np.concatenate((np.ones((N, 1)), x), axis=1)
print(f"tx = {tx}")


#divide dataset
ratio = 0.8
split = int(np.floor(ratio * N))
tx_test = tx[split:]
y_test = y[split:]
tx = tx[:split]
y = y[:split]
N = tx.shape[0]



#train by calling the right function
initial_w = np.zeros(tx.shape[1])
loss, w = implementations.reg_logistic_regression(y, tx, 0.0001, initial_w, 5000, 0.01)

#print(f"Loss -> {loss}")
print(f"final w : {w}")

predictions = implementations.logistic_predict(tx, w)
print(f"predictions = {predictions}")
print(f"and y was : {y}")
errors = np.count_nonzero(predictions - y)
print(f"{errors} errors for {N} samples")
accuracy = 1. - errors / N
print(f"Accuracy : {100. * accuracy}%")
print(f"y 1 count was {y.sum()}, so predicting only 0 yields {100. * (1. - y.sum() / N)}% accuracy")

#predict? visualize?
import matplotlib
utils.line_and_scatter_plot(y, tx, predictions)


#testing accuracy
N_test = y_test.shape[0]
test_predictions = implementations.logistic_predict(tx_test, w)
test_errors = np.count_nonzero(test_predictions - y_test)
test_accuracy = 1. - test_errors / N_test
print(f"\n\n=============================== TESTING ===============================")
print(f"{test_errors} errors for {N_test} samples")
print(f"Accuracy : {100. * test_accuracy}%")
print(f"y 1 count was {y_test.sum()}, so predicting only 0 yields {100. * (1. - y_test.sum() / N_test)}% accuracy")

#write to csv
