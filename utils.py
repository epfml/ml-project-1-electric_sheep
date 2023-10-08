import numpy as np
import matplotlib.pyplot as plt

#there's 321 features in the dataset
def load_data(x_dataset_path, y_dataset_path, max_rows=None):
    x = np.genfromtxt(x_dataset_path, delimiter=",", skip_header=1, max_rows=max_rows)
    y = np.genfromtxt(y_dataset_path, delimiter=",", skip_header=1, max_rows=max_rows)
    # TODO use converters for text data. maybe missing_values/filling_values
    # converter example : converters={0: lambda x: 0 if b"Male" in x else 1},
    # TODO maybe normalize the dataset (also convert to sensible unit if they're american), 
    # extrapolate for the data we don't have, stuff like that

    return x, y


#==========================Plotting==========================#
def scatter_plot():
    ...

def line_plot():
    ...

def line_and_scatter_plot(y, tx, w):
    plt.scatter(tx[:,0], y, c='r')
    plt.plot(tx[:, 0], tx @ w)
    plt.show()


#====================Generate Random Data====================#
def generate_linear_data_with_gaussian_noise(N, d) :

    transform = np.random.rand((d)) * 2 - 1 # linear function mapping a d long feature vector on a number
    #print(f"transform = {transform}")

    X_f = 10.0
    X = np.random.normal(size=(N, d)) * X_f

    noise_f = 6
    noise = np.random.normal(size=N) * noise_f * np.linalg.norm(transform)

    y = transform @ X.T + noise # labels

    return y, X