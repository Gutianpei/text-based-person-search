'''training'''

#import torchvision
import numpy as np
import tensorflow as tf
from dataset import DataGenerator
from keras.layers import Conv2D, MaxPooling2D, Concatenate,Flatten, Input, Dense, Dropout, InputLayer, LSTM, Bidirectional,Dot, Add, Subtract,BatchNormalization
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras import optimizers
from keras.utils import plot_model
# from resnet50 import run_resnet50
from keras.utils import plot_model
from gensim.models import KeyedVectors
import keras
import matplotlib.pyplot as plt
from keras import backend as K
import pickle
import json
# from utils.loss import triplet_loss_adapted_from_tf, cos_distance
from keras.optimizers import Adam
from keras import regularizers
#
# GPU config
config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)
#
#
#
#
#
# load images and captions
dpath = "../datasets/CUHK-PEDES/"
train_path = "caption_train.json"
val_path = "caption_val.json"
IMG_HEIGHT = 384
IMG_WIDTH = 128
batch_size = 128
train_data = json.load(open(train_path))
val_data = json.load(open(val_path))

params = {'batch_size': 64,
          'height': IMG_HEIGHT,
          'width': IMG_WIDTH,
          'shuffle': True,
          'dataset_path': dpath,
          }

word_model = KeyedVectors.load_word2vec_format('word_model.bin')
train_gen = DataGenerator(train_data, word_model,  **params)
val_gen = DataGenerator(val_data, word_model, **params)

###### Resnet ########
resnet = ResNet50(include_top = False, weights = 'imagenet', input_shape=(384,128,3))
output = resnet.layers[-1].output
#output = Dense(1024)
resnet = Model(resnet.input, output=output)
for layer in resnet.layers:
    layer.trainable = False

##### Model #############
img_in = Input(shape=(384,128,3))
res_in = resnet(img_in)
res_conv = Conv2D(512,(1,1),activation='relu')(res_in)
# res_conv2 = Conv2D(512,(1,1),activation='relu')(res_conv)
res_pool = MaxPooling2D(pool_size = (3,3))(res_conv)
res_flat = Flatten()(res_pool)  #shape: n*12*4*2048 -> n*(12*4*2048)
res_nn = Dense(1024, kernel_regularizer = regularizers.l2(0.),activation = 'linear')(res_flat)

cap_in = Input(shape=(50,50))
bi_lstm = Bidirectional(LSTM(20, return_sequences=True))(cap_in)
cap_flat = Flatten()(bi_lstm)
cap_nn = Dense(1024,kernel_regularizer = regularizers.l2(0.),activation = 'linear')(cap_flat)

#BN here?

inner = Dot(axes=1, normalize=True)([res_nn, cap_nn])

base_model = Model(input = [img_in,cap_in], output = inner)
print(base_model.summary())
#plot_model(base_model, to_file='base_model.png', show_shapes = True, show_layer_names = True)
########

########
pos_img = Input(shape = (384,128,3))
pos_cap = Input(shape=(50,50))
neg_img = Input(shape = (384,128,3))
neg_cap = Input(shape=(50,50))

pos_s = base_model([pos_img, pos_cap])
neg_s1 = base_model([neg_img, pos_cap])
neg_s2  = base_model([pos_img, neg_cap])

up = Subtract()([neg_s1,pos_s])
low = Subtract()([neg_s2,pos_s])

dense_up = Dense(1,activation = 'relu', trainable=False, bias_initializer = keras.initializers.Constant(0.2),kernel_initializer=keras.initializers.Ones())(up)
dense_low = Dense(1,activation = 'relu', trainable=False, bias_initializer = keras.initializers.Constant(0.2),kernel_initializer=keras.initializers.Ones())(low)

out = Add()([dense_up,dense_low])

model = Model(input = [pos_img, pos_cap, neg_img, neg_cap], output = out)


#plot_model(model, to_file='model.png', show_shapes = True, show_layer_names = True)

model.compile(loss="mean_squared_error",
             optimizer = "adam",
             metrics = ['mse'])

history = model.fit_generator(generator = train_gen,
                              epochs=3,
                              validation_data=val_gen,
                              # use_multiprocessing = True,
                              # workers = 4,
                              verbose=1)
model.save('default1.h5')

# print("Training Accuracy: " + str(history.history['cosine_proximity'][-1]))
# print("Testing Accuracy: " + str(history.history['val_cosine_proximity'][-1]))
#   # Visualize
# plot_model(model, to_file='model.png')
# plt.plot(history.history['cosine_proximity'])
# plt.plot(history.history['val_cosine_proximity'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
