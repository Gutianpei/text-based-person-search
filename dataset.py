'''Construct dataset'''
import os
import torch
import random
import json
import numpy as np
from torch.utils import data
import cv2

class dataset(data.Dataset):
    def __init__(self, dataset_dir, new_height, new_width):
        self.dataset_dir = dataset_dir
        self.new_height = new_height
		self.new_width = new_width

    def __getitem__(self,index):
        js_data = json.load(self.dataset_dir + "/captions_all.json")

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

    def __len__(self):
        return len(captions)
