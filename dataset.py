'''Construct dataset'''
import os
import torch
import random
import json
import numpy as np
import cv2

class dataset():
    def __init__(self, dataset_dir, new_height, new_width):
        self.dataset_dir = dataset_dir
        self.new_height = new_height
		self.new_width = new_width

    def get_img_caption(self):
        js_data = json.load(self.dataset_dir + "/captions_all.json")
        # Check captions.json for format
        images = []
        captions = []
        for item in js_data:
            #Read image
            image = cv2.imread("/imgs" + item["file_path"])
            image = cv2.resize(image, (self.new_width, self.new_height))
            image = image[:,:,::-1]
            images.append(image.numpy())

            #Read captions
            #TODO!


        images = np.array(images, np.float32)
        images = torch.from_numpy(images).float()

        return images, captions
