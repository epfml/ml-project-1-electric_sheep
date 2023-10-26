import utils
import implementations
import weights

import matplotlib.pyplot as plt

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

"""
    THINGS WE CAN DO: (TODO)
    - devise optimal feature choice and processing (consider removing features with very small weight associated) maybe add some features
    - automatically find optimal lambda, learning rates, etc. 
    - rewrite least_squares using np.linalg.solve (and maybe also ridge_regression)
    - find the optimal c with gradient descent or binary search
    - if a sample has more than f fraction of missing values, drop it

    and maybe:
    - finish poly feature expansion (only if we can make scalar values work)
    - extrapolate for the data we don't have, stuff like that
    - remove outliers 
    - have some command to interrupt training and test using the weights we got so far  
    - maybe write the settings to a log file, so when we have a very good result we can remember them to be able to reproduce it
"""

# feature selection

# with scalar features
x_features = [0, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 88, 89, 96, 97, 98, 104, 105, 108, 109, 110, 117, 118, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 145, 147, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 197, 199, 200, 201, 202, 203, 204, 205, 206, 215, 216, 249, 252, 254, 267, 268, 269, 270, 271, 272, 277, 278, 292, 293, 296, 297, 303, 304, 305]
#x_features = [0, 1, 15, 16, 17, 18, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 88, 89, 92, 96, 97, 98, 99, 100, 101, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 219, 249, 252, 254, 267, 268, 269, 270, 271, 272, 277, 278, 292, 293, 296, 297, 303, 304, 305]
d = len(x_features) - 1
c = np.full(d, True)
c[-18:] = False # All the scalar features

#without them
#usecols = [26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 50, 51, 52, 58, 60, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 87, 88, 95, 96, 97, 103, 104, 107, 108, 109, 116, 117, 127, 128, 129, 131, 132, 133, 134, 135, 136, 137, 138, 140, 141, 142, 144, 146, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 196, 198, 199, 200, 201, 202, 203, 204, 205, 214, 215]
#x_features = [0, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 88, 89, 96, 97, 98, 104, 105, 108, 109, 110, 117, 118, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 145, 147, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 197, 199, 200, 201, 202, 203, 204, 205, 206, 215, 216]
#c = np.full(len(x_features) - 1, True)

#with only them
#x_features = [0, 249, 252, 254, 267, 268, 269, 270, 271, 272, 277, 278, 292, 293, 296, 297, 303, 304, 305]
#x_features = [0, 249]
#c = np.full(len(x_features) - 1, False)

# load the data
x_train, x_test, y_train, train_ids, test_ids = utils.load_data(
    x_train_path="dataset/x_train.csv", 
    y_train_path="dataset/y_train.csv", 
    x_test_path="dataset/x_test.csv", 
    max_rows_train=150000, 
    max_rows_test=None, 
    x_features=x_features
)
#indices = np.random.permutation(x_train.shape[0])[:100000]
#x_train = x_train[indices]
#y_train = y_train[indices]

# Data Processing #NOTE idea: if there's a lot of data processing, we could do it once and save it to a new .csv file, and use this as dataset to train faster
def all_x_processing(x, c):

    x_scalar = x[:, ~c]
    x_scalar = utils.replace_missing_features_with_mean(x_scalar)
    tx_scalar = utils.build_poly(x_scalar, 6)
    tx_scalar[:, 1:] = utils.normalize(tx_scalar[:, 1:]) # we shouldn't normalize the first column (1 vector)

    x_categorical = x[:, c]
    x_categorical = utils.one_hot_encoding(x_categorical)

    return np.concatenate((tx_scalar, x_categorical), axis=1)


# problem : tx_train and tx_test might not contain samples from all possible categories for all features. therefore, one_hot_encoding may return inconsistent results.
# solution : either we concatenate x_train and x_test and do all the processing on it. that would be very sensible, and make the mean-replace and normalisation more robust.

N_train = x_train.shape[0]
x = np.concatenate((x_train, x_test), axis=0)
tx = all_x_processing(x, c)
print(f"After all the processing, tx shape = {tx.shape}")
tx_train = tx[:N_train]
tx_test = tx[N_train:]

#divide dataset
ratio = 0.8
tx_train_train, tx_train_test, y_train_train, y_train_test = utils.split_data(ratio, tx_train, y_train)

N_train_train = tx_train_train.shape[0]
d = tx_train_train.shape[1]


#train by calling the right function
initial_w = weights.weights_449
print(f"initial_w shape = {initial_w.shape}")
#loss, w = implementations.reg_logistic_regression(y_train_train, tx_train_train, 0, initial_w, 2000, 0.3)
#loss, w = implementations.reg_logistic_regression_sgd(y_train_train, tx_train_train, 0, initial_w, 20000, 0.3, 1024, 0.7)
loss, w = implementations.reg_logistic_regression_adam(y_train_train, tx_train_train, 1e-7, initial_w, 10, 0.0001, 2048, 0.9, 0.999)

#print(f"Loss -> {loss}")
print(f"\n\n=============================== OBTAINED WEIGHTS ===============================\n\n")
print(f"{w}\n\n")


#training accuracy
cs = [0.1, 0.2, 0.25, 0.3, 0.4]
print(f"\n\n=============================== TRAINING RESULTS ===============================")
for c in cs:
    print(f"when c = {c}:")
    utils.evaluate(tx_train_train, w, y_train_train, c=c)






#testing accuracy
print(f"\n\n=============================== TESTING RESULTS ===============================")
f1s = []
for c in cs: 
    print(f"when c = {c}: ", end='')
    f1 = utils.evaluate(tx_train_test, w, y_train_test, c=c)
    f1s.append(f1)

best_c = np.argmax(f1s)
print(f"Best cutoff c : {cs[best_c]}, yields f1 = {f1s[best_c]}")

c_array = []
f1_array = []

#for c in np.arange(0, 1, 0.05):
#    c_array.append(c)
#    f1_array.append(utils.evaluate(tx_train_test, w, y_train_test, c=c))
#plt.plot(c_array, f1_array, c='r')
#plt.show()

# generate submission testing data
#for c in cs:
#    test_predictions = implementations.logistic_predict(tx_test, w, c=c) # should be y values between 0 and 1
#    test_predictions = test_predictions * 2 - 1 # should now be between -1 and 1 as desired
#    utils.create_csv_submission(test_ids, test_predictions, f"optimal_submission.csv")

optimal_c, optimal_f1 = utils.find_optimal_c(tx_train_test, y_train_test, w)
print(f"\n\n========Found Optimal c = {optimal_c}, yielding f1 {optimal_f1}========\n")
test_predictions = implementations.logistic_predict(tx_test, w, c=optimal_c) # should be y values between 0 and 1
test_predictions = test_predictions * 2 - 1 # should now be between -1 and 1 as desired

#write to csv
utils.create_csv_submission(test_ids, test_predictions, f"optimal_submission_{optimal_f1}.csv")

threshold = 0.005
print(f"w dimension = {w.shape}")
print(f"w_trunc dimension = {np.sum(np.abs(w) > threshold)}")
w_trunc = w.copy()
w_trunc[np.abs(w) <= threshold] = 0.
optimal_c_trunc, optimal_f1_trunc = utils.find_optimal_c(tx_train_test, y_train_test, w_trunc)
print(f"\n\n========TRUNCATED W -> Found Optimal c = {optimal_c_trunc}, yielding f1 {optimal_f1_trunc}========\n")

if optimal_f1 > optimal_f1_trunc:
    utils.create_csv_submission(test_ids, test_predictions, f"optimal_submission_{optimal_f1}.csv")
else:
    test_predictions = implementations.logistic_predict(tx_test, w, c=optimal_c_trunc) # should be y values between 0 and 1
    test_predictions = test_predictions * 2 - 1 # should now be between -1 and 1 as desired
    utils.create_csv_submission(test_ids, test_predictions, f"optimal_submission_{optimal_f1_trunc}.csv")