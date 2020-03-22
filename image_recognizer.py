import numpy as np
import cv2

import model.conv2D as model

class ImageRecognizer:

  def __init__(self, 
              cascade_configuration_path, 
              model_weights_path, 
              dataset_directory_path, 
              image_size, 
              capture_video_path=None):
    self._dataset_directory_path = dataset_directory_path
    self._model = model.ConvModel(model_weights_path, dataset_directory_path, image_size)
    self.cap = self._make_video_capture(capture_video_path)
    self.classifier = cv2.CascadeClassifier(cascade_configuration_path)
    self._image_size = image_size

  def _make_video_capture(self, source_file_path=None):
    """ Returns a VideoCapture object using a video file if any, otherwise uses the webcam """
    if source_file_path is None:
      # Capture video from webcam. 
      cap = cv2.VideoCapture(0)
      if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
      # Use a video file as input 
      cap = cv2.VideoCapture(source_file_path)

    return cap

  def _detect_target_images(self, frame, classifier):
    """ Returns the images that contains the desired target according the cascade classifier passed """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    target_images = classifier.detectMultiScale(gray, 1.3, 5)

    return target_images

  def _extract_targets_from_images(self, frame, target_images):
    """ Returns an image array containing the target from the images pased by parameter """
    img_array = []
    for (x,y,w,h) in target_images:
      # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # to mark the face but without cropping
      img_array.append(frame[y:y+h, x:x+w])

    return img_array

  def _identify_target(self, image):
    return self._model.identify(image)

  def _ask_for_name(self, target):
    name = input('What\'s your name? ')
    return name

  def _save_new_target(self, image, size, output_path, classname):
    """ Save images in drive's output path """
    timestamp = str(datetime.datetime.now().strftime("%d%m%Y_%H-%M-%S"))
    resized_image = cv2.resize(image, size)
    cv2.imwrite(output_path + '/' + classname + '/' + timestamp + '.jpg', resized_image)

  def _train_new_name(self, target, classname):
    self._model.retrain(target, self._dataset_directory_path, classname)

  def _greet(self, name):
    print('Hi ' + name)

  def run(self):
    print('Face Detector running...')
    while(True):
      # Capture frame-by-frame
      ret, frame = self.cap.read()

      if ret == True:
        targets = self._detect_target_images(frame, self.classifier)
        if isinstance(targets, np.ndarray):
          targets = self._extract_targets_from_images(frame, targets)
          for target in targets:
            target = cv2.resize(target, self._image_size)
            name = self._identify_target(target)
            if name:
              self._greet(name)
            else:
              new_name = self._ask_for_name(target)
              self._greet(new_name)
              self._train_new_name(target, new_name)

      # Stop if escape key is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  ImageRecognizer(cascade_configuration_path='./haarcascades/haarcascade_frontalface_alt.xml',
                  model_weights_path='./model/model_weights.h5',
                  dataset_directory_path='./dataset/',
                  image_size=(50, 50),
                  capture_video_path=None).run()
