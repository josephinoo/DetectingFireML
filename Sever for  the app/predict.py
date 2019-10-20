import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import keras

from keras.models import load_model
from keras.utils import CustomObjectScope

from keras.initializers import glorot_uniform






longitud, altura = 150, 150
modelo = './modelo.h5'
pesos_modelo = './pesos.h5'
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    cnn = load_model(modelo)
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("pred: Fires")
  elif answer == 1:
    print("pred: Normal")
  elif answer == 2:
    print("pred: Smoke")

  return answer
predict("uploads/fires.jpeg")

