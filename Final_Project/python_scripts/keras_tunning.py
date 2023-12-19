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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.losses import BinaryCrossentropy

def build_model(hp):
    # create model object
    model = keras.Sequential([
    #adding first convolutional layer
    keras.layers.Conv1D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=3,
        strides=2,
        activation='relu',
        input_shape=(540,3)),

    #  2 convolutional layer
    keras.layers.Conv1D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
        kernel_size=3,
        strides=2,
        activation='relu'),

    #  3 convolutional layer
    keras.layers.Conv1D(
        filters=hp.Int('conv_3_filter', min_value=32, max_value=128, step=16),
        kernel_size=3,
        strides=2,
        activation='relu'),

    keras.layers.Flatten(),
    keras.layers.Dense(1),
])

    #compilation of model
    logits_loss = BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3,1e-4,1e-5])),
              loss=logits_loss,
              metrics=['accuracy'])
    return model


early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3)

#importing random search
import keras_tuner
import keras

#creating randomsearch object
tuner = keras_tuner.RandomSearch(build_model,
                    objective='val_loss',
                    max_trials = 100,
                    directory = 'keras_tuner',
                    project_name = 'run_1')
# search best parameter
tuner.search(X_train,y_train,epochs=20, verbose = 1,
            validation_split=0.1, callbacks=[early_stopping_cb])

