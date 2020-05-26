'''Construct dataset'''
import os
import cv2
import random
import json
import numpy as np
from gensim.models import KeyedVectors
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence
from keras.applications.resnet50 import preprocess_input
import numpy as np
import keras
import tensorflow as tf

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, json_data, word_model, tokenizer, height, width, dataset_path, time_step, batch_size=32,
                 shuffle=True):
        'Initialization'
        self.json_data = json_data
        self.word_model = word_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.shuffle = shuffle
        self.dataset_path = dataset_path
        self.time_step = time_step

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.json_data) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch


        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        json_temp = [self.json_data[k] for k in indexes]

        # Generate data
        pos_img, pos_cap, ids = self.__data_generation(json_temp)
        y = ids

        return [pos_img, pos_cap], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.json_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, json_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        imgs = np.empty((self.batch_size, self.height, self.width, 3))
        caps = []
        ids = np.empty((self.batch_size, 2, 1024))
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, data in enumerate(json_temp):
            # Store sample
            image = cv2.imread(self.dataset_path + "/imgs/" + data["file_path"])
            image = cv2.resize(image, (self.width, self.height))
            image = image[:,:,::-1] #BGR to RGB

            imgs[i,] = image

            ids[i,]= np.full((2,1024),data["id"])


            # gen caps
            caption = data['captions']
            #tokenizer = RegexpTokenizer(r'\w+')
            #tokens = [j.lower() for j in tokenizer.tokenize(caption)]
            #word_model = KeyedVectors.load_word2vec_format('word_model.bin')
            #caps.append(np.array([self.word_model[i] for i in tokens]))

            # BERT Method
            caps.append(self.tokenizer.encode(caption))

        input_ids = sequence.pad_sequences(caps, maxlen=self.time_step, dtype='int', padding='post', truncating='post', value=0)
        input_ids = tf.constant(input_ids)
        attention_mask = np.where(input_ids != 0, 1, 0)
        attention_mask = tf.constant(attention_mask)
        outputs = self.word_model(input_ids, attention_mask=attention_mask)
        caps = np.array(outputs[0])
        #print(caps.shape)
        return imgs, caps, ids
