'''training'''

#import torchvision
import numpy as np
import dataset
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, LSTM
from keras.models import Sequential, Mode
from keras import optimizers
from keras.applications.resnet50 import ResNet50
import keras
# load images and captions
dpath = "../datasets/CUHK-PEDES"
IMG_HEIGHT = 256
IMG_WIDTH = 128
# https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38
data = dataset.dataset(dataset_dir = dpath, new_width = IMG_WIDTH, new_height = IMG_HEIGHT)
images= data.get_img()

print(images.shape)
input_shape = images.shape[1:]
###### Renset50 ############
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, output=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()


##### Model #############
model = Sequential()
# model.add(Bidirectional(LSTM(512)))
model.add(restnet)
model.add(Dense(512, activation='relu', input_shape=input_shape))
model.add(Dropout(0.3))


model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
# history = model.fit_generator(train_generator,
#                               steps_per_epoch=100,
#                               epochs=100,
#                               validation_data=val_generator,
#                               validation_steps=50,
#                               verbose=1)
