import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet import preprocess_input

class ConvModel:

  def __init__(self):
    super().__init__()

  def __make_train_data_gen(self, directory, batch_size, target_size, class_names_list):
    # The 1./255 is to convert from uint8 to float32 in range [0,1].
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    training_data = image_generator.flow_from_directory(directory=directory,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        target_size=target_size,
                                                        classes = class_names_list)

    return training_data

  def __make_model(self):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1] , 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()

    return model

  def __compile_model(self, model):
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

  def fit(self):
    self.model = self.__compile_model(self.__make_model())
    history = model.fit(train_data_gen, steps_per_epoch=1000, epochs=10)

  def identify(self, image):
    if self.model:
      print('identifying image')
      return 'classname'
    else:
      self.fit()
      print('identifying image')
      return 'classname'
