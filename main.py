import utils
import implementations

import numpy as np

# load the data
x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv", max_rows=500, usecols=[27, 28, 62, 63, 251, 252, 270])

#print(f"x : {x}")
#print(f"y : {y}")
x = np.nan_to_num(x)
#print(f"x : {x}")
x = utils.normalize(x)
print(f"means = {np.mean(x, axis=0)}, std = {np.std(x, axis=0)}")
#print(f"x : {x}")

tx = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
print(f"tx = {tx}")



#train by calling the right function
initial_w = np.ones(tx.shape[1]) * 0.1
_, initial_loss = implementations.compute_MSE_gradient_and_loss(y, tx, initial_w)
loss, w = implementations.mean_squared_error_gd(y, tx, initial_w, 10000, 1)
#loss, w = implementations.least_squares(y, tx)

print(f"Loss : before -> {initial_loss}, after -> {loss}")
print(f"final w : {w}")

predictions = tx @ w
print(f"predictions = {predictions}")
predictions = np.where(predictions > 0.5, np.ones_like(predictions), np.zeros_like(predictions))
print(f"predictions integrated : {predictions}")
print(f"and y was : {y}")

#predict? visualize?
import matplotlib
utils.line_and_scatter_plot(y, tx, w)
