'''Script for testing
'''
from keras.models import load_model
import numpy as np
from testing_data import get_test
from collections import Counter
from gensim.models import KeyedVectors
import time
import tqdm
from keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from keras import backend as K
from numba import jit
from triplet_loss import batch_hard_triplet_loss

@jit(nopython=True)
def compute_score(mat, ids):
	rank1 = 0
	rank5 = 0
	rank10 = 0
	rank20 = 0

	idx = 0 # Keep track of true id
	print()
	print("Computing RankX")
	for i in range(len(mat)):
		idx_score = mat[i]
		res_ids = ids[idx_score]

		match = False
		for i, res_id in enumerate(res_ids):
			if res_id == ids[idx]:
				match = True
			if i == 0 and match:
				rank1 += 1
			if i == 9 and match:	# we have replicated image, so here i is 9
				rank5 += 1
			if i== 19 and match:
				rank10 += 1
			if i == 39:
				if match > 0:
					rank20 += 1
				break
		idx += 1

	print("Rank1: ")
	print(rank1/(idx+1))
	print("Rank5: ")
	print(rank5/(idx+1))
	print("Rank10: ")
	print(rank10/(idx+1))
	print("Rank20: ")
	print(rank20/(idx+1))

def get_models(model):
	print(model.summary())
	print("Image Weights: ")
	img_input = model.layers[0].input
	img_output = model.layers[-3].output
	img_model = Model(img_input, img_output)
	print(img_model.get_weights())

	print("Caption Weights: ")
	cap_input = model.layers[-6].input
	cap_output = model.layers[-2].output
	cap_model = Model(cap_input, cap_output)
	print(cap_model.get_weights())
	exit()
	return img_model, cap_model

def triplet_loss(y_true, y_pred):

    label = y_true[:,0,0]

    loss = batch_hard_triplet_loss(label, y_pred[:,1], y_pred[:,0], 0.2)
    return loss

word_model = KeyedVectors.load_word2vec_format('word_model.bin')
TIME_STEP = 50
ids, imgs, caps = get_test("caption_test.json", "../datasets/CUHK-PEDES", word_model, TIME_STEP)

model = load_model("../best_model.h5", custom_objects={'tf': tf, 'triplet_loss': triplet_loss, 'K': K})

img_model, cap_model = get_models(model)	# get img path and cap path and resemble to new models

print("data and model loaded")

print("Computing Distance")
img_out = img_model.predict(imgs)
cap_out = cap_model.predict(caps)
mat = cosine_similarity(cap_out, img_out) #compute cosine, output shape(6248*6248) --> (cap, img)
mat = np.array([np.argsort(score)[::-1] for score in mat])
mat = mat.astype(np.int)
ids = ids.astype(np.int)
print("Matrix Ready")

compute_score(mat, ids)
