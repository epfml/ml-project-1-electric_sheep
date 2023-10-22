import utils
import implementations

import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def compare(p):
    (_, loss) = p
    return loss

# TODO : do the same with poly expansion

losses = []
for i in range(321):

    x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv", max_rows=1000, usecols=[i])
    x = x.reshape((x.shape[0], 1))

    full_rows = np.logical_not(np.isnan(x).any(axis=1))
    x = x[full_rows]
    y = y[full_rows]
    x = utils.normalize(x)
    tx = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)

    initial_w = np.ones(tx.shape[1])
    loss, w = implementations.logistic_regression(y, tx, initial_w, 200, 0.001)

    print(f"loss for feature {i} : {loss}")
    losses.append((i, loss))

losses.sort(key=compare)
print(f"\n\nLosses:\n {losses}")
