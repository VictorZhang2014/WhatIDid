# -*- coding: utf-8 -*-
"""transfer_learning_tutorial.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13XQtteu358m5AP65tmBItQPUpaL_53Af

Tutorial from  https://keras.io/applications/

# Plot Probilities
"""

import matplotlib.pyplot as plt

def plot_bar(predictions):
    types = [pred[1] for pred in predictions]
    probs = [pred[2] for pred in predictions]
    
    plt.barh(np.arange(len(probs)), probs)
    _ = plt.yticks(np.arange(3), types)
    plt.show()

from google.colab import files

uploaded = files.upload()

"""# Load image as digit matrix"""

from keras.preprocessing import image
import numpy as np

img_path = 'elephant.jpeg'
def load_img(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

"""# ResNet50"""

from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

def predict_byResNet50():
  model = ResNet50(weights='imagenet')

  x = load_img(img_path)
  x = preprocess_input(x)
  preds = model.predict(x)
  # decode the results into a list of tuples (class, description, probability)
  # (one such list for each sample in the batch)

  predictions = decode_predictions(preds, top=3)[0]
  print('Predicted: {}'.format(predictions))
  plot_bar(predictions)
  
predict_byResNet50()

"""# VGG 16"""

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input, decode_predictions

def predict_byVGG16():
  model = VGG16(weights='imagenet')

  x = load_img(img_path)
  x = preprocess_input(x)
  preds = model.predict(x)

  predictions = decode_predictions(preds, top=3)[0]
  print('Predicted: {}'.format(predictions))
  plot_bar(predictions)
  
predict_byVGG16()

"""# VGG 19"""

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions

def predict_byVGG19():
  model = VGG19(weights='imagenet')

  x = load_img(img_path)
  x = preprocess_input(x)
  preds = model.predict(x)

  predictions = decode_predictions(preds, top=3)[0]
  print('Predicted: {}'.format(predictions))
  plot_bar(predictions)
  
predict_byVGG19()

"""# InceptionV3"""

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions

def predict_byInceptionV3():
  model = InceptionV3(weights='imagenet')

  x = load_img(img_path)
  x = preprocess_input(x)
  preds = model.predict(x)

  predictions = decode_predictions(preds, top=3)[0]
  print('Predicted: {}'.format(predictions))
  plot_bar(predictions)
  
predict_byInceptionV3()

"""# MobileNet"""

from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions

def predict_byMobileNet():
  model = MobileNet(weights='imagenet')

  x = load_img(img_path)
  x = preprocess_input(x)
  preds = model.predict(x)

  predictions = decode_predictions(preds, top=3)[0]
  print('Predicted: {}'.format(predictions))
  plot_bar(predictions)
  
predict_byMobileNet()

"""# Xception"""

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input, decode_predictions

def predict_byXception():
  model = Xception(weights='imagenet')

  x = load_img(img_path)
  x = preprocess_input(x)
  preds = model.predict(x)

  predictions = decode_predictions(preds, top=3)[0]
  print('Predicted: {}'.format(predictions))
  plot_bar(predictions)
  
predict_byXception()

