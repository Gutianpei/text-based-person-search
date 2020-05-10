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

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, json_data, word_model, height, width, dataset_path, time_step, batch_size=32,
                 shuffle=True):
        'Initialization'
        self.json_data = json_data
        self.word_model = word_model
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
            ids[i,].fill(data["id"])
            # gen caps
            caption = data['captions']
            tokenizer = RegexpTokenizer(r'\w+')
            tokens = [j.lower() for j in tokenizer.tokenize(caption)]
            #word_model = KeyedVectors.load_word2vec_format('word_model.bin')
            caps.append(np.array([self.word_model[i] for i in tokens]))
        caps = sequence.pad_sequences(caps, maxlen=self.time_step, dtype='float', padding='pre', truncating='pre', value=0.0)
        #print(caps.shape)
        neg_img = np.roll(imgs, 12, axis=0)
        neg_cap = np.roll(caps, 12, axis=0)
        return imgs, caps, neg_img, neg_cap
# class dataset():
#     def __init__(self, dataset_dir, new_height, new_width):
#         self.dataset_dir = dataset_dir
#         self.new_height = new_height
#         self.new_width = new_width
#         # self.n = n
#
#     def get_data(self):
#         js_data = json.load(open(self.dataset_dir + "/caption_all.json"))
#         # Check captions.json for format
#         images = []
#         captions = []
#         for item in js_data:
#             #Read image
#             #print(os.getcwd())
#             image = cv2.imread(self.dataset_dir + "/imgs/" + item["file_path"])
#             image = cv2.resize(image, (self.new_width, self.new_height))
#             #print(image)
#             image = image[:,:,::-1] #BGR to RGB
#             images.append(image)
#             images.append(image)
#             # if len(images) == 8:
#             #      break
#
#         #Read captions
#         vector_size = 50
#
#         #js_data = json.load(open(self.dataset_dir + "/caption_all.json"))
#         captions = [i['captions'] for i in js_data]
#         flatten_caps = sum(captions, [])
#
#         # tokenize
#         tokenizer = RegexpTokenizer(r'\w+')
#         tokens = [[j.lower() for j in tokenizer.tokenize(i)] for i in flatten_caps]
#
#         # word embedding training (could change to pre-trained one)
#         word_model = Word2Vec(tokens, size=vector_size, min_count=1)
#         word_model.wv.save_word2vec_format('word_model.bin')
#
#         # convert
#         caption_embeddings = []
#         for i in tokens:
#             temp_emb = []
#             for j in i:
#                 temp_emb.append(word_model[j])
#             caption_embeddings.append(np.array(temp_emb))
#             # if len(caption_embeddings) == 8:
#             #      break
#
#         caption_embeddings = sequence.pad_sequences(caption_embeddings, maxlen=50, dtype='float', padding='post', truncating='post', value=0.0)
#
#         return np.array(images), np.array(caption_embeddings)
