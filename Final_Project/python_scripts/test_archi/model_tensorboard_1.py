import numpy as np
def load_split_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
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
    X_train = npz[k[0]]
    y_train = npz[k[1]]
    X_test = npz[k[2]]
    y_test = npz[k[3]]
    X_val = npz[k[4]]
    y_val = npz[k[5]]

    print("Train:", X_train.shape)
    print("Label Train:", y_train.shape)
    print("Test:", X_test.shape)
    print("Label Test:", y_test.shape)
    print("Validation:", X_val.shape)
    print("Label Validation:", y_val.shape)

    return X_train, y_train, X_test, y_test, X_val, y_val

path ='/home/users/h/henrymi2/seisbench_gpu/data/new_era_split.npz'
X_train, y_train, X_test, y_test, X_val, y_val = load_split_data(path)

import os
file_path = '/home/users/h/henrymi2/seisbench_gpu'
model = 'model1'
def get_CNN_logdir():
    time = np.datetime64('now').astype(str)
    run_logdir = os.path.join(file_path,"CNN_logs", f"{model}_{time[:-3]}") # time goes in the fstring
    return run_logdir
print(get_CNN_logdir())

def get_checkpoint_logdir():
    time = np.datetime64('now').astype(str)
    checkpoint_logdir = os.path.join(file_path,"Modelcheckpoint_logs", f"{model}_{time[:-3]}") # time goes in the fstring
    return checkpoint_logdir
print(get_checkpoint_logdir())

import tensorflow as tf
from tensorflow import keras

model = tf.keras.models.Sequential([

    # Convolution 1
    keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation="relu", input_shape=(540,3)),
    # output( 64, 269)
    keras.layers.BatchNormalization(),

    ## Convolution 2
    #keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation="relu"),
    ## output( 64, 134)
    #keras.layers.BatchNormalization(),

    # # Convolution 3
    # keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation="relu"),
    # # output( 64, 66)
    # keras.layers.BatchNormalization(),

    # # Convolution 4
    # keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation="relu"),
    # # output( 64, 32)
    # keras.layers.BatchNormalization(),

    # # Convolution 5
    # keras.layers.Conv1D(filters=64, kernel_size=3, strides=2, activation="relu"),
    # # output( 64, 16)
    # keras.layers.BatchNormalization(),

    keras.layers.Flatten(),
    keras.layers.Dense(1),
])

model.build()
model.summary()

from keras.losses import BinaryCrossentropy

# from keras.optimizers import Adam
# adam_opt = Adam(learning_rate=1e-5)

logits_loss = BinaryCrossentropy(from_logits=True) 

model.compile(loss=logits_loss, optimizer='adam', metrics=['accuracy'])

# early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(get_checkpoint_logdir(),
                                                   save_best_only=True,
                                                   monitor= 'val_loss')
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_CNN_logdir())

model.reset_metrics()
keras.backend.clear_session()

model.fit(X_train, y_train, epochs=50, batch_size=2**6, verbose=0, validation_split=0.1,
          callbacks=[
              # early_stopping_cb,
              checkpoint_cb,
              tensorboard_cb])
