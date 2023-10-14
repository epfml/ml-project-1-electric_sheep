import utils
import implementations

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

# load the data
x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv", max_rows=1000, usecols=[27, 28, 62, 63, 251, 252, 270])

# Data Processing #TODO idea: if there's a lot of data processing, I could do it once and save it to a new .csv file, and use this as dataset to train faster
x = utils.feature_specific_processing(x)
x = utils.replace_missing_features_with_mean(x)
#x = utils.normalize(x)
tx = x
tx = utils.build_poly(x, 2)
#tx = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
print(f"tx = {tx}")



#train by calling the right function
initial_w = np.zeros(tx.shape[1])
loss, w = implementations.reg_logistic_regression(y, tx, 0, initial_w, 4000, 0.0001)

#print(f"Loss -> {loss}")
print(f"final w : {w}")

predictions = implementations.logistic_predict(tx, w)
print(f"predictions = {predictions}")
print(f"and y was : {y}")
errors = np.count_nonzero(predictions - y)
print(f"{errors} errors for {x.shape[0]} samples")

#predict? visualize?
import matplotlib
utils.line_and_scatter_plot(y, tx, predictions)
