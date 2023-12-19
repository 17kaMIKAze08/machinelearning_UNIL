import numpy as np
def load_npz(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from an npz file (file containing multiples arrays and not compressed)

    Parameters:
    - file_path: The path to the npz file.

    Returns:
    - Tuple containing arrays loaded from the npz file.
        X has shape (batch_size, data_dim)
        Y has shape (batch_size,)
    """
    # Load data from npz file and displaying keys
    npz = np.load(file_path)
    print("Loading data from:", file_path)
    k = []
    for keys in npz.files:
        print("Available key in NPZfile:", keys)
        k.append(keys)

    # Extract arrays from the loaded data
    X = npz[k[0]]
    Y = npz[k[1]]

    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    return X, Y

path = '/home/users/h/henrymi2/seisbench_gpu/new_era.npz'
X, Y = load_npz(path)

def preprocess(x: np.ndarray) -> np.ndarray:
    """
    Preprocess the input matrix for Keras.

    Parameters:
    - x: Input data

    Returns:
    - Normalised and Transposed data ready for Keras.
    """
    # Normalize the data and removing the mean
    Xnorm = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True))
    print("Standard Deviation after normalization:", Xnorm.std(axis=0))

    # Transpose the data for Keras
    Xt = np.transpose(Xnorm, axes=(0, 2, 1))

    print("Standard Deviation after transpose:", Xt.std(axis=0))
    print("Preprocessed data shape:", x.shape)
    print("Processed data shape:", Xt.shape)

    return Xt

Xt = preprocess(X)

from sklearn.model_selection import train_test_split
X_train, x_temp, y_train, y_temp = train_test_split(Xt , Y, test_size=0.3)
X_test, X_val, y_test, y_val = train_test_split(x_temp , y_temp, test_size=0.2)

print("Train:", X_train.shape)
print("Label Train:", y_train.shape)
print("Test:", X_test.shape)
print("Label Test:", y_test.shape)
print("Validation:", X_val.shape)
print("Label Validation:", y_val.shape)

del X, Xt

from sklearn.model_selection import train_test_split
def save_train_test_split_data(X_train, y_train, X_test, y_test, X_val, y_val, name):
    split_file = f"/home/users/h/henrymi2/seisbench_gpu/{name}.npz"

    # Save the split data in a NumPy file
    np.savez(split_file, train_X=X_train, 
             train_Y=y_train, test_X=X_test, 
             test_y=y_test, val_X=X_val, 
             val_y=y_val)

save_train_test_split_data(X_train, y_train, X_test, y_test, X_val, y_val, name="new_era_split")
