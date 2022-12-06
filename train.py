import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

def get_frames(filename: str) -> np.ndarray:
  
  # get image frames

  # return:
  #   frames: np.ndarray (90, 21, 3)
  data = []
  with open(filename,'rb') as f:
    data = pickle.load(f)
  return data[:90,:]

# for every frame --> (21, 3)
# (90, 21, 3)
def model_fn(timesteps: int = 3) -> tf.keras.Model:
  batch_size = 8
  frames = timesteps*30
  data_dim = (21,3)

# expected input data shape: (batch_size, timesteps, data_dim)
  model = keras.Sequential()
  # decrease neuron number accordingly
  model.add(layers.LSTM(units = 128, return_sequences = True, input_shape = (frames, data_dim)))
  #model.add(Dropout(0.3))
  model.add(layers.LSTM(128), return_sequences = True)
  #model.add(Dropout(0.3))
  model.add(layers.Flatten())
  # sigmoid or relu would be function for dense layer
  model.add(layers.Dense(128, activation='relu'))
  # classification layer - sigmoid (0 : bad, 1 : good)
  model.add(layers.Dense(1, activation='sigmoid'))
  #
  return model

model = model_fn()
model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs= 2, batch_size = 32,
                    validation_data=(X_val, Y_val),
                    callbacks=[checkpoint_cb])

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
