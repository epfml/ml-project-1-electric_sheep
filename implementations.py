import numpy as np

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
        gradient, _ = compute_MSE_gradient_and_loss(y, tx, w)
        w = w - gamma * gradient

    _, loss = compute_MSE_gradient_and_loss(y, tx, w)

    return loss, w


# TODO : maybe modularize with GD.
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm.

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

    N = y.shape[0]

    w = initial_w

    for _ in range(max_iters):
        n = np.random.randint(0, N)
        gradient, _ = compute_MSE_gradient_and_loss(y[n], tx[n], w)
        w = w - gamma * gradient

    _, loss = compute_MSE_gradient_and_loss(y, tx, w)

    return loss, w


def least_squares(y, tx):
    ...

def ridge_regression(y, tx, lambda_):
    ...

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    ...

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    ...



# ================== Helpers / Modularized functions ==================

def compute_MSE_gradient_and_loss(y, tx, w):
    """Calculate the gradient and the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, d)
        w: numpy array of shape=(d,). The vector of model parameters.

    Returns:
        A numpy array of shape (d, ) (same shape as w), containing the gradient of the loss at w,
        and the value of the loss (a scalar), corresponding to the input parameters w.
    """
    N = y.shape[0]
    e = y - tx @ w

    loss = e.T @ e / (2*N)
    gradient = - tx.T @ e / N

    return gradient, loss




