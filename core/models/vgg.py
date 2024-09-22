from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import preprocess_input

from .base import BaseClassficiationModel

class VGGClassificationModel(BaseClassficiationModel):
    
    def __init__(self):
        super().__init__()
        self.model = VGG16()
    
    def predict(self, image):
        image = image.resize((224, 224))  # Resize the image
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the VGG model
        image = preprocess_input(image)
        yhat = self.model.predict(image)
        # convert the probabilities to class labels
        label = decode_predictions(yhat)
        # retrieve the most likely result, e.g. highest probability
        label = label[0][0]
        # print the classification
        return '%s (%.2f%%)' % (label[1], label[2]*100)