from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Softmax,GlobalAveragePooling2D, Reshape
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


IMG_HEIGHT = 384
IMG_WIDTH = 128
img_path = "1_1.jpg"
img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


# Resnet
resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = resnet.layers[-1].output
resnet = Model(resnet.input, output=output)
for layer in resnet.layers:
    layer.trainable = False

features = resnet.predict(x)

resneted = []

for feat in features:

    horizontal_six = []
    for i in range(0,6):
        f = feat[i*2:i+2]
        s = f.mean(axis=0)
        horizontal_six.append(s)
    horizontal_six = np.array(horizontal_six)
    resneted.append(horizontal_six)

resneted = np.array(resneted)

print(resneted.shape)
