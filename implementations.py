import numpy as np
import ml_methods

# ==================== the functions to implement ====================

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,d)
        initial_w: numpy array of shape=(d, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the loss value (scalar) corresponding to the w obtained after the last iteration
        w: numpy array of shape=(d, ) (the same as initial_w). The final weight vector obtained after the last training iteration
    """

    w = initial_w

    for _ in range(max_iters):
        gradient, loss = ml_methods.compute_MSE_gradient_and_loss(y, tx, w)
        w = w - gamma * gradient

    _, loss = ml_methods.compute_MSE_gradient_and_loss(y, tx, w)

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm.

    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,d)
        initial_w: numpy array of shape=(d,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: the loss value (scalar) corresponding to the w obtained after the last iteration
        w: numpy array of shape=(d,) (the same as initial_w). The final weight vector obtained after the last training iteration
    """
    N = tx.shape[0]
    w = initial_w

    for _ in range(max_iters):
        n = np.random.randint(N)
        gradient, loss = ml_methods.compute_MSE_gradient_and_loss(y[[n]], tx[[n]], w)
        w = w - gamma * gradient

    _, loss = ml_methods.compute_MSE_gradient_and_loss(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """ Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,d), d is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(d,), d is the number of features.
        mse: scalar., mean squared error
    """

    w_opt = np.linalg.solve(tx.T @ tx, tx.T@y)
    _, loss = ml_methods.compute_MSE_gradient_and_loss(y, tx, w_opt)
    return w_opt, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N, d), d is the number of features.
        lambda_: scalar, the factor controlling the strength of the normalisation

    Returns:
        w: optimal weights, numpy array of shape(d,), d is the number of features.
    """
    N = y.shape[0]
    d = tx.shape[1]
    w_opt = np.linalg.solve(tx.T @ tx + (lambda_ * 2 * N) * np.identity(d), tx.T @ y)
    _, loss = ml_methods.compute_MSE_gradient_and_loss(y, tx, w_opt)
    return w_opt, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ implements the logistic regression classification method, with gradient descent optimisation.

    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N, d)
        initial_w: numpy array of shape (d,), used as starting weights for the iterative optimisation
        max_iters: integer, the amount of iterations performed during optimisation
        gamma: scalar, learning rate for gradient descent
    """

    w = initial_w

    for _ in range(max_iters):
        gradient, loss = ml_methods.compute_cross_entropy_gradient_and_loss(y, tx, w)
        w = w - gamma * gradient

    _, loss = ml_methods.compute_cross_entropy_gradient_and_loss(y, tx, w)

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ implements the logistic regression classification method, with gradient descent optimisation.

    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N, d)
        lambda_: scalar, the factor controlling the strength of the normalisation 
        initial_w: numpy array of shape (d,), used as starting weights for the iterative optimisation
        max_iters: integer, the amount of iterations performed during optimisation
        gamma: scalar, learning rate for gradient descent
    """

    w = initial_w

    for _ in range(max_iters):
        gradient, loss = ml_methods.compute_cross_entropy_gradient_and_loss(y, tx, w)
        gradient += 2 * lambda_*w # regularisation term

        w = w - gamma * gradient

    _, loss = ml_methods.compute_cross_entropy_gradient_and_loss(y, tx, w)

    return w, loss



