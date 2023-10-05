import numpy as np

#there's 321 features in the dataset
def load_data(x_dataset_path, y_dataset_path, max_rows):
    ...
    x = np.genfromtxt(x_dataset_path, delimiter=",", skip_header=1, max_rows=max_rows)
    y = np.genfromtxt(y_dataset_path, delimiter=",", skip_header=1, max_rows=max_rows)
    # TODO use converters for text data. maybe missing_values/filling_values
    # converter example : converters={0: lambda x: 0 if b"Male" in x else 1},
    # TODO maybe normalize the dataset (also convert to sensible unit if they're american), 
    # extrapolate for the data we don't have, stuff like that

    return x, y

