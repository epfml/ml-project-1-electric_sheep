import utils
import implementations

import numpy as np

def test_load_data():
    x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv")
    print(f"x sample : {x[0:10]}")
    print(f"y sample : {y[0:10]}")

    assert y == np.array([0., 0., 0., 0., 0., 0., 1., 1., 0., 0.])

def run_all_tests():
    test_load_data()


run_all_tests()