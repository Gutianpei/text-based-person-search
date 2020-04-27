'''training'''

#import torchvision
import numpy as np
import tensorflow as tf
import dataset
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer, LSTM, Bidirectional
from keras.models import Sequential
from keras import optimizers
from resnet50 import run_resnet50
from keras.utils import plot_model
import keras
import matplotlib.pyplot as plt
from keras import backend as K
import pickle
from utils.loss import triplet_loss_adapted_from_tf, cos_distance
from keras.optimizers import Adam


# GPU config
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)





# load images and captions
dpath = "../datasets/CUHK-PEDES"
IMG_HEIGHT = 384
IMG_WIDTH = 128
batch_size = 32
# https://towardsdatascience.com/deep-learning-using-transfer-learning-python-code-for-resnet50-8acdfb3a2d38
data = dataset.dataset(dataset_dir = dpath, new_width = IMG_WIDTH, new_height = IMG_HEIGHT)

# captions = data.get_caption()
images= data.get_img()

print(images.shape)
input_shape = images.shape[1:]

###### Renset50 ############
print("Resnet running")
img_feat = run_resnet50(images)
# print(img_feat.shape)
n2_feat = np.concatenate((img_feat, img_feat),axis = 0)
# print(captions.shape)
print(n2_feat.shape)
pickle.dump(n2_feat, open("resnet50_avgp.pkl", "wb"))
exit()
##### Model #############
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=captions.shape[1:]))
model.add(Flatten())
model.add(Dense(216, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(1000, activation='softmax'))

print(model.summary())

model.compile(loss=triplet_loss_adapted_from_tf,
              optimizer="adam",
              metrics=['cosine_proximity'])

history = model.fit(captions, n2_feat,
                              epochs=30,
                              batch_size=batch_size,
                              validation_split=0.3,
                              verbose=1)

print("Training Accuracy: " + str(history.history['cosine_proximity'][-1]))
print("Testing Accuracy: " + str(history.history['val_cosine_proximity'][-1]))
  # Visualize
plot_model(model, to_file='model.png')
plt.plot(history.history['cosine_proximity'])
plt.plot(history.history['val_cosine_proximity'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
