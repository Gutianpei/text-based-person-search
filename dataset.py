'''Construct dataset'''
import os
import torch
import random
import json
import numpy as np

class dataset():
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def get_dict(self,index):
        js_data = json.load(self.dataset_dir)   #load json to dict
        return js_data

    def get_img_list(self):
        
