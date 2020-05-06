import json
import random
import copy
import cv2
import numpy as np


# def Diff(li1, li2):
#     return (list(set(li1) - set(li2)))
#
# random.seed(10)
# ids = [i for i in range(1,13004)]
# #print(len(ids))
# test = random.sample(ids, k = 1000)
# #print(len(test))
# train_val = Diff(ids, test)
# #print(len(train_val))
#
# val = random.sample(train_val, k = 1000)
# train = Diff(train_val, val)
# #print(len(train))
#
# #print(Diff(train+val+test,ids))
# #exit()
# test_data = []
# train_data = []
# val_data = []
# js_data = json.load(open("../datasets/CUHK-PEDES/caption_all.json"))
# for data in js_data:
#     if len(data["captions"]) == 2:
#
#         for cap in data["captions"]:
#             d_new = copy.deepcopy(data)
#             d_new["captions"] = cap
#
#             if d_new["id"] in test:
#                 test_data.append(d_new)
#             elif d_new["id"] in val:
#                 val_data.append(d_new)
#
#             else:
#                 train_data.append(d_new)
#
#
# random.shuffle(train_data)
# random.shuffle(test_data)
# random.shuffle(val_data)
#
# with open('caption_test.json', 'w') as outfile:
#     json.dump(test_data, outfile)
#
# with open('caption_train.json', 'w') as outfile:
#     json.dump(train_data, outfile)
#
# with open('caption_val.json', 'w') as outfile:
#     json.dump(val_data, outfile)

def get_test(json_path,dataset_path):
    ''' Read from caption_test.json
        Args:
            json_path:  .../.../caption_test.json
            dataset_path:  .../.../CUHK_PEDES
        Returns:
            ndarray
            imgs: (1000,384,128,3)
            ids: (1000,1)
            caps: (2xxx,50,50)
    '''
    ids = []
    imgs = []
    caps = []

    js_data = json.load(open(json_path))
    for data in js_data:

        image = cv2.imread(dataset_path + "/imgs/" + data["file_path"])
        image = cv2.resize(image, (128, 384))
        image = image[:,:,::-1] #BGR to RGB

        ids.append(js_data["ids"])
        imgs.append(image)

        captions = data["captions"]
        ###TODO


    return np.array(ids),np.array(imgs),np.array(caps)