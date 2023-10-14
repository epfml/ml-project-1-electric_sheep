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

    for n_iter in range(max_iters):
        gradient, loss = compute_MSE_gradient_and_loss(y, tx, w)
        w = w - gamma * gradient

        #print(f"iter {n_iter} : loss = {loss}")

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

    for n_iter in range(max_iters):
        n = np.random.randint(0, N)
        gradient, loss = compute_MSE_gradient_and_loss(y[[n]], tx[[n]], w)
        w = w - gamma * gradient

        print(f"iter {n_iter} : loss = {loss}")

    _, loss = compute_MSE_gradient_and_loss(y, tx, w)

    return loss, w

#TODO : rewrite using np.linalg.solve
def least_squares(y, tx):
    w_opt = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    _, loss = compute_MSE_gradient_and_loss(y, tx, w_opt)
    return loss, w_opt


def ridge_regression(y, tx, lambda_):
    N = y.shape[0]
    d = tx.shape[1]
    w_opt = np.linalg.inv(tx.T @ tx + (lambda_ * 2 * N) * np.identity(d)) @ tx.T @ y
    _, loss = compute_MSE_gradient_and_loss(y, tx, w_opt)
    return loss, w_opt

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        gradient, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)
        w = w - gamma * gradient

        print(f"iter {n_iter} : loss = {loss}")

    _, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)

    return loss, w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        gradient, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)
        w = w - gamma * (gradient + 2*lambda_*w)

        if n_iter % 20 == 0:
            print(f"iter {n_iter} : loss = {loss}")

    _, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)

    return loss, w



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

def compute_cross_entropy_gradient_and_loss(y, tx, w):

    #h = 1. / (1. + np.exp(-tx@w))
    #gradient = (h - y) @ tx
    #loss = -np.sum(y * np.log(h) + (1. - y) * np.log(1. - h)) / y.shape[0]

    N = tx.shape[0]
    b = tx@w
    gradient = tx.T @ (sigmoid(b) - y) / N
    loss = np.sum(np.log(1. + np.exp(b)) - y * b) / N

    return gradient, loss

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def logistic_predict(tx, w):
    sigma = 1. / (1. + np.exp(-tx@w))
    return np.where(sigma > 0.5, np.ones_like(sigma), np.zeros_like(sigma))




