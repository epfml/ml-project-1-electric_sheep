import utils
import implementations

import numpy as np

# load the data
x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv", max_rows=1000, usecols=[27, 28, 62, 63, 251, 252, 270])

# Data Processing #TODO idea: if there's a lot of data processing, I could do it once and save it to a new .csv file, and use this as dataset to train faster
x = utils.replace_missing_features_with_mean(x)
x = utils.normalize(x)
# x = utils.build_poly(x)
tx = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)



#train by calling the right function
initial_w = np.ones(tx.shape[1])
loss, w = implementations.logistic_regression(y, tx, initial_w, 300, 0.1)

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
