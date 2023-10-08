import utils
import implementations

import numpy as np

def test_load_data():
    x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv", max_rows=20)

    x = x[0:10]
    y = y[0:10]

    #print(f"x sample : {x[0:10]}")
    #print(f"y sample : {y[0:10]}")

    assert (y == np.array([0., 0., 0., 0., 0., 0., 1., 1., 0., 0.])).all()
    print("    -load_data test passed!")

def test_compute_MSE_gradient_and_loss():
    #say N=3 and d=2
    y = np.array([1.5, -0.5, 0.3])
    tx = np.array([
        [0.5, 2.],
        [-1., 0.],
        [0.4, 1.2]
    ])
    w = np.array([0.4, -0.6])

    true_gradient = - np.array([1.694, 6.032]) / 3.
    true_loss = (6.25 + 0.01 + 0.7396) / 6.
    #print(f"true_gradient={true_gradient}, true_loss={true_loss}")

    gradient, loss = implementations.compute_MSE_gradient_and_loss(y, tx, w)
    #print(f"gradient={gradient}, loss={loss}")


    assert (np.isclose(gradient, true_gradient)).all() and np.isclose(true_loss, loss)
    print("    -compute_MSE_gradient_and_loss test passed!")


def run_all_tests():
    test_load_data()
    test_compute_MSE_gradient_and_loss()

    print("-> All test passed!!")



run_all_tests()