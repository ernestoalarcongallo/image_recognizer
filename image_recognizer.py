import cv2
import model.convModel as model

class ImageRecognizer:

  def __init__(self, file_path, cascade_configuration_path):
    super().__init__()
    self.cap = self.__make_video_capture(file_path)
    self.classifier = cv2.CascadeClassifier(cascade_configuration_path)

  def __make_video_capture(self, file_path):
    """ Returns a VideoCapture object using a video file if any, otherwise uses the webcam """
    if not file_path:
      # Capture video from webcam. 
      cap = cv2.VideoCapture(0)
      if not cap.isOpened():
        print("Cannot open camera")
        exit()
    else:
      # Use a video file as input 
      cap = cv2.VideoCapture(file_path)

    return cap

  def __detect_target_images(self, frame):
    """ Returns the images that contains the desired target according the cascade classifier passed """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    target_images = self.classifier.detectMultiScale(gray, 1.3, 5)

    return target_images

  def __extract_targets_from_images(self, frame, target_images):
    """ Returns an image array containing the target from the images pased by parameter """
    img_array = []
    for (x,y,w,h) in target_images:
      # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) # to mark the face but without cropping
      img_array.append(frame[y:y+h, x:x+w])

    return img_array

  def __identify_target(self, image):
    print('identifying faces...')
    return model.identify(image)

  def __ask_for_name(self, target):
    name = raw_input('What\'s your name?')
    return name

  def __save_new_target(self, image, size, output_path, classname):
    """ Save images in drive's output path """
    timestamp = str(datetime.datetime.now().strftime("%d%m%Y_%H-%M-%S"))
    resized_image = cv2.resize(image, size)
    cv2.imwrite(outputPath + '/' + classname + '/' + timestamp + '.jpg', resized_image)

  def __train_new_name(self, target, classname):
    model.fit(target, classname)

  def __greet(self, name):
    print('Hi ' + name)

  def run(self):
    print('Face Detector running...')
    while(True):
      # Capture frame-by-frame
      ret, frame = self.cap.read()

      if ret == True:
        targets = self.__detect_target_images(frame, self.classifier)
        if isinstance(targets, np.ndarray):
          targets = self.__extract_targets_from_images(frame, targets)
          for target in targets:
            name = self.__identify_target(target)
            if name:
              self.__greet(name)
            else:
              new_name = self.__ask_for_name(self, target)
              self.__greet(new_name)
              self.__train_new_name(self, target, new_name)

      # Stop if escape key is pressed
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
  ImageRecognizer().run()
