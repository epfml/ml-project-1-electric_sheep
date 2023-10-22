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
        n = np.random.randint(n)
        gradient, loss = compute_MSE_gradient_and_loss(y[[n]], tx[[n]], w)
        w = w - gamma * gradient

        print(f"iter {n_iter} : loss = {loss}")

    _, loss = compute_MSE_gradient_and_loss(y, tx, w)

    return loss, w

def least_squares(y, tx):
    w_opt = np.linalg.solve(tx.T @ tx, tx.T@y)
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
        gradient += 2 * lambda_*w # regularisation term

        w = w - gamma * gradient

        if n_iter % 100 == 0:
            print(f"iter {n_iter} : loss = {loss}")

    _, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)

    return loss, w

def reg_logistic_regression_sgd(y, tx, lambda_, initial_w, max_iters, gamma, batch_size, b=None):
        
    w = initial_w
    m = initial_w if b is not None else None

    for n_iter in range(max_iters):
        batch_y, batch_tx = batch(y, tx, batch_size)
        gradient, loss = compute_cross_entropy_gradient_and_loss(batch_y, batch_tx, w)
        gradient += 2 * lambda_*w

        if b is not None:
            m = b * m + (1.0-b) * gradient
            w = w - gamma * m
        else:
            w = w - gamma * gradient

        #w = w - (gamma * n_iter / max_iters) * gradient

        if n_iter % 10 == 0:
            _, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)
            print(f"iter {n_iter} : loss = {loss}")

    _, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)

    return loss, w

def reg_logistic_regression_adam(y, tx, lambda_, initial_w, max_iters, gamma, batch_size, b1, b2):
    
    w = initial_w
    m = initial_w
    v = initial_w

    for n_iter in range(max_iters):
        batch_y, batch_tx = batch(y, tx, batch_size)
        gradient, loss = compute_cross_entropy_gradient_and_loss(batch_y, batch_tx, w)
        gradient += 2 * lambda_*w

        m = b1 * m + (1 - b1) * gradient
        v = b2 * v + (1 - b2) * (gradient ** 2)

        m_hat = m / (1 - b1 ** (n_iter + 1))
        v_hat = v / (1 - b2 ** (n_iter + 1))

        w = w - gamma * m_hat / (np.sqrt(v_hat) + 1e-8)

        if n_iter % 10 == 0:
            _, loss = compute_cross_entropy_gradient_and_loss(y, tx, w)
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
    h = sigmoid(b)
    gradient = tx.T @ (h - y) / N

    loss = np.sum(np.log(1. + np.exp(b)) - y * b) / N
    #loss = -np.sum(y * np.log(h) + (1. - y) * np.log(1. - h)) / y.shape[0]


    return gradient, loss

def sigmoid(x):
    #return 1. / (1. + np.exp(-x))
    return 0.5 * (1.0 + np.tanh(0.5 * x))

def logistic_predict(tx, w, c=0.5):
    #sigma = 1. / (1. + np.exp(-tx@w))
    sigma = sigmoid(tx@w)
    return np.where(sigma > c, np.ones_like(sigma), np.zeros_like(sigma))

def batch(y, tx, batch_size):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    N = len(y)

    indices = np.random.permutation(np.arange(N))[0:batch_size]
    return y[indices], tx[indices]


