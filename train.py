import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import os
from typing import Optional
def get_data(filepath: str) -> np.ndarray:
    """
    Get data(image frames) of chosen video

    Parameters
    ----------
    filename : str; name of file to open

    Returns
    -------
    frames : np.ndarray, (90, 21, 3)
    """
    with open(filepath,'rb') as f:
        data = pickle.load(f)
    return data

def get_valid_data(filepath: str) -> Optional[np.ndarray]:
    """
    Confirms validity of the chosen data, and return only valid data
    The data is valid only if it has ndarray with shape of (90, 21*3)

    Parameters
    ----------
    filepath: str; path to file

    Returns
    -------
    valid_data: ndarray[Optional]; ndarray(90, 63) if valid, None otherwise
    """
    data = get_data(filepath)
    if data.shape == (90, 21*3):
        return data
    else:
        return None

def get_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate full dataset in a single ndarray from stored data.

    Returns
    -------
    dataset: ndarray(N,90,63); all data in one ndarray with N being number of samples
    results: ndarray(N); correct labels of each data, where 1 is True and 0 is false
    """
    dataset = []
    results = []

    path_frame_true = r".\input_data\Frames\True"
    path_frame_false = r".\input_data\Frames\False"
    paths = (path_frame_true, path_frame_false)
    for path_frame in paths:
        iter = os.scandir(path=path_frame)          # iterates through all files in the path
        for file in iter:
            filename = file.name
            filepath = path_frame + '\\' + filename

            data = get_valid_data(filepath)
            if data is None:                        # confirms if data.shape == (90, 21*3)
                continue
            else:
                dataset.append(data)
                if 'true' in filename:
                    results.append(np.ones(1))
                else:
                    results.append(np.zeros(1))
    
    dataset = np.array(dataset)
    results = np.array(results)
    return dataset, results

def get_model() -> tf.keras.Model:
    """
    Generate model to train. Consist of multiple layers.
    LSTM -> LSTM -> Flatten -> Dense -> Dense
    Constraint: data is prepared for data with ...
    batch_size(# of samples): N, time_steps: 90 frames, input_dim: 21*3

    Returns
    -------
    model: tf.keras.Model; Machine learning model to train
    """
    num_frames = 90                     # each video are taken for 90 frames
    num_landmarks = 21
    num_dim = 3
    num_data = num_landmarks * num_dim

    # expected input data shape: ([batch_size, ]timesteps, data_dim) -> ([num_video, ]frames, 21*3)
    model = keras.Sequential()
    # decrease neuron number accordingly
    model.add(layers.LSTM(units = 128, return_sequences = True, input_shape = (num_frames, num_data)))
    #model.add(Dropout(0.3))
    model.add(layers.LSTM(units = 128))
    #model.add(Dropout(0.3))
    model.add(layers.Flatten())
    # sigmoid or relu would be function for dense layer
    model.add(layers.Dense(128, activation='relu'))
    # classification layer - sigmoid (0 : bad, 1 : good)
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def set_model(epochs: int = 2, isSaved: bool = False) -> tf.keras.Model:
    """
    Generate model and save it to directory. Also returns the model.

    Parameters
    ----------
    epochs: int[Optional]; epochs for model fit process. Default is 2
    isSaved: bool; if true, saves the model

    Returns
    -------
    model: tf.keras.Model
    """
    X_train, Y_train = get_dataset()

    model = get_model()
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(x = X_train, y = Y_train, batch_size = len(X_train), epochs= epochs)

    if isSaved:
        model.save('./model')
    return model

def test_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> int:
    """
    Perform tests on the model.

    Parameters
    ----------
    model: tf.keras.Model; Model to test on
    X_test: ndarray(N, 90, 21*3); test dataset
    y_test: ndarray(N, 1); test labels

    Returns
    -------
    accuracy: double; accuracy of the model in percentage, rounded at 100th digit
    """
    # model = tf.keras.models.load_model('./model')
    X_test.reshape(-1, 90, 21*3)
    y_test.reshape(-1, 1)
    scores = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print("Test accuracy: %.2f%%" % (scores[1]*100))
    return round(scores[1]*100, 2)

if __name__ == '__main__':
    """
    Repeat the process for {cnt} times.
    Find accuracy of the model with given train_set and test_set at different epochs(1~50).
    Stored in pkl file as ndarray(2,50)
    """
    for cnt in range(7):
        results = [range(1, 51), []]
        for epoch in range(1, 51):
            ### new module ###
            model = set_model(epochs=epoch, isSaved=False)
            ### loaded module ###
            # model = tf.keras.models.load_model('./model')

            path_frame_true = r".\test_set\True"
            path_frame_false = r".\test_set\False"
            paths = (path_frame_true, path_frame_false)

            ##### automatic test_set test #####
            test_set = []
            test_label = []
            for path_frame in paths:
                iter = os.scandir(path=path_frame)          # iterates through all files in the path
                for file in iter:
                    filename = file.name
                    filepath = path_frame + '\\' + filename

                    data = get_valid_data(filepath)
                    if data is None:                        # confirms if data.shape == (90, 21*3)
                        continue
                    else:
                        test_set.append(data)
                        if 'true' in filename:
                            test_label.append(np.ones(1))
                        else:
                            test_label.append(np.zeros(1))
            
            test_set = np.array(test_set)
            test_label = np.array(test_label)
            acc = test_model(model, test_set, test_label)
            results[1].append(acc)
    
        with open(f'./statistics/accuracy_{cnt}','wb') as f:
            pickle.dump(np.array(results), f)