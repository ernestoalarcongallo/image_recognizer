import os
from datetime import datetime

import cv2
import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class ConvModel:

  def __init__(self, model_weights_path, dataset_directory_path, image_size):
    self._dataset_directory_path = dataset_directory_path
    self._image_size = image_size
    self._model_weights_path = model_weights_path
    try:
      self._categories = self._load_categories(dataset_directory_path)
      self._model = self._make_compiled_model(image_size, self._categories)
      self._model.load_weights(model_weights_path)
    except:
      self._create_and_save_new_model(dataset_directory_path, image_size, model_weights_path)


  def _load_normalized_dataset(self, directory):
    X, y, categories = [], [], []

    for folder in os.listdir(directory):
      if not folder.startswith('.'):
        categories.append(folder)
        index = categories.index(folder)
        for file in os.listdir(os.path.join(directory, folder)):
          y.append(index)
          X.append(cv2.imread(os.path.join(directory, folder, file)))
    
    X, y = np.array(X), np.array(y)
    X = X.astype('float32') / 255

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=123)
    y_train_one_hot = to_categorical(y_train, num_classes=len(categories))
    y_test_one_hot = to_categorical(y_test, num_classes=len(categories))

    return X_train, y_train_one_hot, X_test, y_test_one_hot, categories


  def _load_categories(self, directory):
    categories = []
    for folder in os.listdir(directory):
      if not folder.startswith('.'):
        categories.append(folder)

    return categories


  def _make_compiled_model(self, image_size, categories):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size[0], image_size[1], 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(categories), activation='softmax'))
    model.summary()               
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


  def _create_and_save_new_model(self, dataset_directory_path, image_size, model_path):
    X_train, y_train_one_hot, X_test, y_test_one_hot, categories = self._load_normalized_dataset(dataset_directory_path)
    self._model = self._make_compiled_model(image_size, categories)
    self._model.fit(X_train, y_train_one_hot, batch_size=32, epochs=20, validation_split=0.2)
    self._model.save(model_path)


  def retrain(self, image, dataset_directory, classname):
    timestamp = str(datetime.now().strftime("%d%m%Y_%H-%M-%S"))

    try:
      os.mkdir(dataset_directory)
      cv2.imwrite(dataset_directory + '/' + classname + '/' + timestamp + '.jpg', image)
    except FileExistsError:
      cv2.imwrite(dataset_directory + '/' + classname + '/' + timestamp + '.jpg', image)

    self._create_and_save_new_model(self._dataset_directory_path, self._image_size, self._model_weights_path)


  def identify(self, image):
    prediction = ''
    probabilities = self._model.predict(np.array([image]).astype('float32'))
    for probability in probabilities:
      if np.amax(probabilities) > 0.98:
        prediction = self._categories[np.argmax(probabilities)]

    return prediction
