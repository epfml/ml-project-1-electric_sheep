import numpy as np
import matplotlib.pyplot as plt

import csv

import ml_methods

#===========================Data Pre-Processing===========================#

#there's 321 features in the dataset
def load_data(x_train_path=None, y_train_path=None, x_test_path=None, max_rows_train=None, max_rows_test=None, x_features=None):
    """
    This function loads the data and returns the respectinve numpy arrays.

    Args:
        x_train_path, y_train_path, x_test_path (str): datafolder paths to the three datasets
        max_rows_train, max_rows_test : (int) The amount of rows we read in the train and test datasets
        x_features : (array[int]) The index of the columns we read in the file, i.e. the features we use

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        y_train_path,
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
        max_rows=max_rows_train
    ) if y_train_path is not None else None

    x_train = np.genfromtxt(
        x_train_path, 
        delimiter=",", 
        skip_header=1, 
        max_rows=max_rows_train, 
        usecols=x_features
    ) if x_train_path is not None else None

    x_test = np.genfromtxt(
        x_test_path, 
        delimiter=",", 
        skip_header=1, 
        max_rows=max_rows_test, 
        usecols=x_features
    ) if x_test_path is not None else None

    train_ids = x_train[:, 0].astype(dtype=int) if x_train_path is not None else None
    test_ids = x_test[:, 0].astype(dtype=int) if x_test_path is not None else None
    x_train = x_train[:, 1:] if x_train_path is not None else None
    x_test = x_test[:, 1:] if x_test_path is not None else None

    y_train = (y_train + 1) / 2 # put y between 0 and 1

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def normalize(x):
    """
    A function to normalize an array, returning the data with 0 mean and 1 std_dev along each column
   
    Args:
        x : A (N, d) shaped array containing heterogeneous data
        c: A (d,) shaped boolean array indicating which variables are categorical

    Returns:
        A (N, d) shaped array (like x), with mean 0 and std_dev 1 along each column
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

def rows_with_all_features(x):
    """Finds the rows with all features non nan in x
    """
    missing_elems = np.isnan(x)
    return np.logical_not(missing_elems.any(axis=1))

def remove_rows_with_missing_features(x, y):
    """Remove rows with at least one missing features in x and y, i.e. returns new x and y's without those rows
    """

    full_rows = rows_with_all_features(x)
    x = x[full_rows]
    y = y[full_rows]

    return x,y

def replace_missing_features_with_mean(x):
    """Replace all nan values in x with the mean of their column
    """
    return np.nan_to_num(x, nan=np.nanmean(x, axis=0))


def build_poly(x, degree, bias=True):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N, d), N is the number of samples and d the number of features
        degree: integer.

    Returns:
        numpy array of shape (N, d'), where d' is the total number of expanded features.
    """

    N = x.shape[0]
    to_concat = []
    if(bias):
        to_concat.append(np.ones((N,1)))
    for i in range(1, degree+1):
        to_concat.append(x ** i)

    return np.concatenate(to_concat, axis=1)



def one_hot_encoding(x):
    """
        Args:
            x: Numpy array of shape (N, d) with some categorical features, can contain NaN values
        Returns:
            A numpy array of shape (N, d'), where each of the categorical feature was converted to one-hot encoding
    """

    if x.shape[1] == 0:
        return x

    N = x.shape[0]
    d = x.shape[1]
    # we assume x only contains categorical features
    x = np.nan_to_num(x, nan=0)
    x = x.astype('int32')

    for i in range(d):
        _, collapsed = np.unique(x[:, i], return_inverse=True)
        # suppose we have col = [7, 5, 5, 7, 5], then, we obtain categories = [5, 7], collapsed = [1, 0, 0, 1, 0]
        x[:, i] = collapsed


    cat_counts = np.nanmax(x, axis=0)+1 # cat_counts is a (d,) shaped array with the max value for each feature
    d_prime = np.sum(cat_counts, dtype=np.int32)
    z = np.zeros((N, d_prime))
    # we don't want to substract the first, we want to slide everything : [1, 3, 2] -> [1, 4, 6] -> [0, 1, 4]
    cum_cat_counts = np.roll(np.cumsum(cat_counts), 1)
    cum_cat_counts[0] = 0
    indexes = x + cum_cat_counts
    z[np.arange(N), indexes.T] = 1

    return z

def split_data(ratio, tx, y):
    """Splits tx and y in 2 according to ratio.
    This is used for validation.
    The train sets contain a fraction ratio of the original data,
    and the test set contain a fraction (1- ratio) of it.
    """

    split = int(np.floor(ratio * tx.shape[0]))
    tx_test = tx[split:]
    y_test = y[split:]
    tx_train = tx[:split]
    y_train = y[:split]

    return tx_train, tx_test, y_train, y_test


#==========================Evaluating==========================#

def evaluate(tx, w, y, c=0.5, print_=True):
    """
    Computes, and potentially displays, the validation results.
    The metric of choice is the f1-score, giving a balanced importance to the true positive and the true negatives.
    """
    N = tx.shape[0]
    y_pred = ml_methods.logistic_predict(tx, w, c)

    correct = y_pred == y
    pos = y_pred == 1

    n_correct = np.sum(correct)
    accuracy = n_correct / N

    p = pos.sum()
    tp = (correct & pos).sum()
    fp = p - tp
    tn = (~pos & correct).sum()
    fn = (N - p) - tn

    f1 = 2 * tp / (2 * tp + fp + fn)

    if print_:
        print(f"    {N - correct.sum()} errors for {N} samples")
        print(f"    Accuracy : {100. * accuracy}%")
        print(f"    F1 : {f1}")

    return f1



def find_optimal_c(tx, y, w):
    """
    Finds the optimal cutoff parameter to be used in the logistic regression prediction.
    Logistic regression predicts probabilities, and the prediction process decides whether to predict 1 or 0 according to those probabilities.
    We predict using a cut-off between 0 and 1, and predict 1 if the probability is higher than the cut-off.
    In order to maximise the f1-score, the optimal cut-off, isn't always 0.5 (which maximises the accuracy).
    Assuming the f1-score is concave in the cut-off and approximately smooth, we find the optimal one with a simple grid search approach.
    We empirically found that the optimal cut-off is almost always between 0.1 and 0.3, so we only consider those values.
    """

    best_f1 = 0
    best_c = 0
    for c in np.arange(0.1, 0.3, 0.001):
        f1 = evaluate(tx, w, y, c=c, print_=False)
        if f1 > best_f1:
            best_f1 = f1
            best_c = c

    print(f"final best c : {best_c} yielding f1={best_f1}")
    return best_c, best_f1


#==========================Plotting==========================#
def plot_f1_to_c(tx_train_test, w, y_train_test, delta=0.01):
    """
    Draws a simple plot to see how the f1-score changes with the cut-off c
    """
    c_array = []
    f1_array = []
    for c in np.arange(0, 1, delta):
        c_array.append(c)
        f1_array.append(evaluate(tx_train_test, w, y_train_test, c=c, print_=False))
    plt.plot(c_array, f1_array, c='r')
    plt.show()


#====================Generate Random Data====================#
def generate_linear_data_with_gaussian_noise(N, d) :
    """
    Small util used to test functions with random data
    """
    transform = np.random.rand((d)) * 2 - 1 # linear function mapping a d long feature vector on a number

    X_f = 10.0
    X = np.random.normal(size=(N, d)) * X_f

    noise_f = 6
    noise = np.random.normal(size=N) * noise_f * np.linalg.norm(transform)

    y = transform @ X.T + noise # labels

    return y, X

