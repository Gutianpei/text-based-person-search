'''Construct dataset'''
import os
import cv2
import random
import json
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence
from keras.applications.resnet50 import preprocess_input

class dataset():
    def __init__(self, dataset_dir, new_height, new_width):
        self.dataset_dir = dataset_dir
        self.new_height = new_height
        self.new_width = new_width

    def get_img(self):
        js_data = json.load(open(self.dataset_dir + "/caption_all.json"))
        # Check captions.json for format
        images = []
        captions = []
        for item in js_data:
            #Read image
            #print(os.getcwd())
            image = cv2.imread(self.dataset_dir + "/imgs/" + item["file_path"])
            image = cv2.resize(image, (self.new_width, self.new_height))
            #print(image)
            image = image[:,:,::-1] #BGR to RGB
            images.append(image)
            print(len(images))
            #Read captions
            #TODO!

        #return images, captions
        return np.array(images)

    def get_caption(self):
        vector_size = 300

        js_data = json.load(open(self.dataset_dir + "/caption_all.json"))
        captions = [i['captions'] for i in js_data]
        flatten_caps = sum(captions, [])

        # tokenize
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = [[j.lower() for j in tokenizer.tokenize(i)] for i in flatten_caps]

        # word embedding training (could change to pre-trained one)
        word_model = Word2Vec(tokens, size=vector_size, min_count=1)
        word_model.wv.save_word2vec_format('word_model.bin')

        # convert
        caption_embeddings = []
        for i in tokens:
            temp_emb = np.zeros(vector_size)
            for j in i:
                temp_emb += word_model[j]
            caption_embeddings.append(temp_emb/len(i))

        return np.array(caption_embeddings)
