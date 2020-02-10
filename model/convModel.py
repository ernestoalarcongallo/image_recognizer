import pickle
import PIL
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import img_to_array

class ConvModel:

  def __init__(self, model_weights_path, dataset_directory_path, batch_size, image_size, classes):
    try:
      self.__model = self.__make_compiled_model(image_size)
      self.__model.load_weights(model_weights_path)
    except:
      self.__create_and_save_new_model(dataset_directory_path, 
                                      batch_size, 
                                      image_size, 
                                      classes,
                                      model_weights_path)

  def __make_train_data_gen(self, directory, batch_size, target_size, classes):
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    training_data_gen = image_generator.flow_from_directory(directory=directory,
                                                            batch_size=batch_size,
                                                            shuffle=True,
                                                            target_size=target_size,
                                                            classes = classes)

    return training_data_gen

  def __make_compiled_model(self, image_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1] , 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

  def __create_and_save_new_model(self, dataset_directory_path, batch_size, image_size, classes, model_path):
    model = self.__make_compiled_model(image_size)
    train_data_gen = self.__make_train_data_gen(dataset_directory_path, 
                                                  batch_size, 
                                                  image_size, 
                                                  classes)
    history = model.fit(train_data_gen, steps_per_epoch=1000, epochs=10)
    self.__model = model
    self.__model.save(model_path)

  def retrain(self, image, dataset_directory, classname):
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    image.save(dataset_directory + '/' + timestamp + '.jpg')

  def identify(self, image):
    predictions = ''
    try:
      image_dtype = tf.image.convert_image_dtype(image, dtype=tf.float16, saturate=False)
      predictions = self.__model.predict(image)
      print('predictions ==>', predictions)
    except ValueError:
      print(ValueError)
      predictions = 'Just an error'

    return predictions
