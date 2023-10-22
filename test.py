import utils
import implementations

import numpy as np

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

def test_mean_squared_error_gd_or_sgd(stochastic=False):

    (N, d) = (1000, 1)

    y, X = utils.generate_linear_data_with_gaussian_noise(N, d)

    initial_w = np.zeros(d)

    #print(f"xshape = {X.shape}, y.shape = {y.shape}, w.shape = {initial_w.shape}")
    loss, w = implementations.mean_squared_error_sgd(y, X, initial_w, 100, 0.002) if stochastic else implementations.mean_squared_error_gd(y, X, initial_w, 30, 0.01)

    #print(f"final loss : {loss}")
    #print(f"final w : {w}")
    utils.line_and_scatter_plot(y, X, w)
    print(f"    -mean_squared_error_{'sgd' if stochastic else 'gd'} test passed!")

def test_least_squares():

    (N, d) = (1000, 1)

    y, X = utils.generate_linear_data_with_gaussian_noise(N, d)

    #print(f"xshape = {X.shape}, y.shape = {y.shape}, w.shape = {initial_w.shape}")
    loss, w = implementations.least_squares(y, X)

    print(f"     ls final loss : {loss}")
    print(f"     w shape = {w.shape}")
    #print(f"final w : {w}")
    #utils.line_and_scatter_plot(y, X, w)
    print(f"    -least_squares test passed!")

def test_ridge_regression():

    (N, d) = (1000, 1)

    y, X = utils.generate_linear_data_with_gaussian_noise(N, d)
    lambda_ = 1.0

    loss, w = implementations.ridge_regression(y, X, lambda_)

    print(f"     rr final loss : {loss}")
    #print(f"final w : {w}")
    utils.line_and_scatter_plot(y, X, w)
    print(f"    -ridge_regression test passed!")

def test_load_data():
    x, y = utils.load_data("dataset/x_train.csv", "dataset/y_train.csv", max_rows=20)

    x = x[0:10]
    y = y[0:10]

    #print(f"x sample : {x[0:10]}")
    #print(f"y sample : {y[0:10]}")

    assert (y == np.array([0., 0., 0., 0., 0., 0., 1., 1., 0., 0.])).all()
    print("    -load_data test passed!")

def test_replace_missing_features_with_mean():
    x = np.array([
        [6.,        np.nan,     2.],
        [np.nan,    np.nan,     3.],
        [2.,        1.,         1.]
    ])

    x_f = utils.replace_missing_features_with_mean(x)
    x_true = np.array([
        [6.,    1.,    2.],
        [4.,    1.,    3.],
        [2.,    1.,    1.]
    ])

    assert np.array_equal(x_f, x_true)
    print("    -replace_missing_features_with_mean test passed!")

def test_one_hot_encoding_old():
    x = np.array([[1, 2], [2, 0], [3, 1]])
    true_ohe = np.array([
        [0, 1, 0, 0, 0, 0, 1], 
        [0, 0, 1, 0, 1, 0, 0], 
        [0, 0, 0, 1, 0, 1, 0]
    ])
    ohe = utils.one_hot_encoding_old(x)
    print(f"ohe = {ohe}")

    assert np.array_equal(ohe, true_ohe)
    print("    -one_hot_encoding_old test passed!")

def test_one_hot_encoding():
    # cats = [3, 5, 7] for feature 1 and [4, 6, 8] for feature 2. x.shape = [4, 2]
    x = np.array([[3, 6], [7, 8], [5, 4], [5, 6]])
    # we want collapsed = [[0, 1], [2, 2], [1, 0], [1, 1]]
    # and then, one hot = [[1 0 0 0 1 0] [0 0 1 0 0 1] [0 1 0 1 0 0], [0 1 0 0 1 0]]
    ohe = utils.one_hot_encoding(x)
    true_ohe = np.array([[1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1], [0, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0]])

    print(f"ohe = {ohe}")
    print(f"true_ohe = {true_ohe}")

    assert np.array_equal(ohe, true_ohe)
    print("    -one_hot_encoding test passed!")

def run_all_tests():
    test_load_data()
    test_compute_MSE_gradient_and_loss()
    test_mean_squared_error_gd_or_sgd(stochastic=False)
    test_mean_squared_error_gd_or_sgd(stochastic=True)
    test_least_squares()
    test_ridge_regression()

    print("-> All test passed!!")



#run_all_tests()

test_least_squares()
#test_ridge_regression()
#test_replace_missing_features_with_mean()
#test_one_hot_encoding()