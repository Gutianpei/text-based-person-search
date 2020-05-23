import json
import random
import copy
import cv2
import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence
from collections import Counter

def data_gen():
    ids = {}
    js_data = json.load(open("caption_train_balanced.json"))

    for data in js_data:
        id = data["id"]
        #print(id)
        if id not in ids:
            ids[id] = [copy.deepcopy(data)]
        else:
            ids[id].append(copy.deepcopy(data))

    balanced_train = []
    print(len(ids))
    for id in ids:
        if len(ids[id]) !=4:
            print(id)
        # # r = min(len(ids[id]),4)
        # # if r!=4:
        # #     print(r)
        #     for i in range(4):
        #         balanced_train.append(copy.deepcopy(ids[id][i]))
    exit()
    print(len(balanced_train))

    with open('caption_train_balanced.json', 'w') as outfile:
          json.dump(balanced_train, outfile)
    # def Diff(li1, li2):
    #     return (list(set(li1) - set(li2)))
    # #
    # random.seed(5)
    #
    # ids = [i for i in range(1,13004)]
    # # #print(len(ids))
    # test = random.sample(ids, k = 1000)
    # # #print(len(test))
    # train_val = Diff(ids, test)
    # # #print(len(train_val))
    # #
    # val = random.sample(train_val, k = 1000)
    # train = Diff(train_val, val)
    # # #print(len(train))
    # #
    # # #print(Diff(train+val+test,ids))
    # # #exit()
    # test_data = []
    # train_data = []
    # val_data = []
    #
    #
    #
    #
    #
    # for data in js_data:
    #     if len(data["captions"]) == 2:
    # #
    #         for cap in data["captions"]:
    #
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
    # #
    # with open('caption_test.json', 'w') as outfile:
    #      json.dump(test_data, outfile)
    # #
    # with open('caption_train.json', 'w') as outfile:
    #      json.dump(train_data, outfile)
    # #
    # with open('caption_val.json', 'w') as outfile:
    #      json.dump(val_data, outfile)

def get_test(json_path, dataset_path, word_model, tokenizer, time_step):
    ''' Read from caption_test.json
        Args:
            json_path:  .../.../caption_test.json
            dataset_path:  .../.../CUHK_PEDES
        Returns:
            ndarray
            imgs: (2000,384,128,3)
            ids: (2000,1)
            caps: (2000,50,50)
    '''
    ids = []
    imgs = []
    caps = []

    js_data = json.load(open(json_path))
    for data in js_data:
        if len(ids) == 1000:
            break
        image = cv2.imread(dataset_path + "/imgs/" + data["file_path"])
        image = cv2.resize(image, (128, 384))
        image = image[:,:,::-1] #BGR to RGB

        ids.append(data["id"])
        imgs.append(image)

        caption = data['captions']
        #tokenizer = RegexpTokenizer(r'\w+')
        #tokens = [j.lower() for j in tokenizer.tokenize(caption)]
        #caps.append(np.array([word_model[i] for i in tokens]))
        #print(len(caps))

        #BERT
        input_ids = tf.constant(tokenizer.encode(caption))[None, :]
        outputs = word_model(input_ids)
        embedding = np.array(outputs[0])
        caps.append(embedding.reshape(-1,768))
    caps = sequence.pad_sequences(caps, maxlen=time_step, dtype='float', padding='pre', truncating='pre', value=0.0)

    return np.array(ids),np.array(imgs),np.array(caps)

def main():
    data_gen()

if __name__ == '__main__':
    main()
