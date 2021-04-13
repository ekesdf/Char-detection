import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image


img = np.asarray(Image.open('/home/yolo/Schreibtisch/custom_model/classification/input images/5.jpg').resize((100,100),Image.NEAREST))
img = np.reshape(img,(1,100,100,3))

reconstructed_model = keras.models.load_model("/home/yolo/Schreibtisch/custom_model/classification/first model/trained_model")


prediction = reconstructed_model.predict(img)

# class1,class2,class3,class4,class5 = prediction
print(prediction)
print(tf.nn.softmax(prediction[0]))

