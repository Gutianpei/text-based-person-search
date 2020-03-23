'''Construct dataset'''
import os
from keras.preprocessing import image as img_util
from keras.applications.resnet50 import preprocess_input
import random
import json
import numpy as np

class dataset():
    def __init__(self, dataset_dir, new_height, new_width, ratio):
        self.dataset_dir = dataset_dir
        self.new_height = new_height
        self.new_width = new_width
        self.ratio = ratio
        
    def get_img_caption(self):
        js_data = json.load(self.dataset_dir + "/captions_all.json")
        # Check captions.json for format
        data_len = len(js_data)
        ratio_iter = self.ratio*data_len
        images = []
        captions = []
        for idx, item in enumerate(js_data):
            if idx < data_len:
                img_path = self.dataset_dir + "/imgs/" + item["file_path"]
                img = image.load_img(img_path, target_size=(self.new_width, self.new_height))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                images.append(x)
                #Read image with pytorch
    #            image = cv2.imread(self.dataset_dir + "/imgs/" + item["file_path"])
    #            image = cv2.resize(image, (self.new_width, self.new_height))
    #            image = image[:,:,::-1]
    #            images.append(image.numpy())

                #Read captions
                #TODO!


#        images = np.array(images, np.float32)
        images = preprocess_input(images)

        return images, captions
