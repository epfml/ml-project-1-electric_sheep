import utils
import implementations

import matplotlib.pyplot as plt

import numpy as np


# feature selection
x_features = [0, 27, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 59, 61, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 77, 88, 89, 96, 97, 98, 104, 105, 108, 109, 110, 117, 118, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 141, 142, 143, 145, 147, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 193, 194, 195, 197, 199, 200, 201, 202, 203, 204, 205, 206, 215, 216, 249, 252, 254, 267, 268, 269, 270, 271, 272, 277, 278, 292, 293, 296, 297, 303, 304, 305]
d = len(x_features) - 1
c = np.full(d, True)
c[-18:] = False # All the scalar features

# load the data
x_train, x_test, y_train, train_ids, test_ids = utils.load_data(
    x_train_path="dataset/x_train.csv", 
    y_train_path="dataset/y_train.csv", 
    x_test_path="dataset/x_test.csv", 
    max_rows_train=5000, 
    max_rows_test=10, 
    x_features=x_features
)

def all_x_processing(x, c, degree):

    x_scalar = x[:, ~c]
    x_scalar = utils.replace_missing_features_with_mean(x_scalar)
    tx_scalar = utils.build_poly(x_scalar, degree)
    tx_scalar[:, 1:] = utils.normalize(tx_scalar[:, 1:]) # we shouldn't normalize the first column (1 vector)

    x_categorical = x[:, c]
    x_categorical = utils.one_hot_encoding(x_categorical)

    return np.concatenate((tx_scalar, x_categorical), axis=1)

N_train = x_train.shape[0]
ratio = 0.8

degrees = [7]
lambdas = [0, 1e-8, 1e-7, 1e-6, 1e-4]
print("Degree/Lambda    0       1e-8    1-e-7   1-e6    1e-4")
results = [] # stores pairs (w, f1)
for degree in degrees:
    print(f"{degree}            ", end='')
    for lambda_ in lambdas:

        x = np.concatenate((x_train, x_test), axis=0)
        tx = all_x_processing(x, c, degree)
        tx_train = tx[:N_train]
        tx_test = tx[N_train:]

        #divide dataset
        tx_train_train, tx_train_test, y_train_train, y_train_test = utils.split_data(ratio, tx_train, y_train)

        N_train_train = tx_train_train.shape[0]
        d = tx_train_train.shape[1]

        #train by calling the right function
        initial_w = np.zeros(d)
        loss, w = implementations.reg_logistic_regression_adam(y_train_train, tx_train_train, lambda_, initial_w, 400, 0.0001, 512, 0.9, 0.999)

        #testing accuracy
        optimal_c, optimal_f1 = utils.find_optimal_c(tx_train_test, y_train_test, w)
        print(f"{optimal_f1:.3f}    ", end='')
        results.append((w, optimal_f1, optimal_c))
    print()

# in the end, we save to some file the best results
optimal_c = None
optimal_w = None
optimal_f1 = 0.
i = 0
opt_idx = 0
for (w, f1, c) in results:
    if f1 > optimal_f1:
        optimal_f1 = f1
        optimal_c = c
        optimal_w = w
        opt_idx = i
    i +=1

print(f"\nfound the best value at idx {opt_idx}") # we can do modulo to find the hpp
test_predictions = implementations.logistic_predict(tx_test, optimal_w, c=optimal_c)
test_predictions = test_predictions * 2 - 1
utils.create_csv_submission(test_ids, test_predictions, f"hpp_optimal_submission.csv")