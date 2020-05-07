'''Script for testing
'''
from keras.models import load_model
import numpy as np
from testing_data import get_test
from collections import Counter
from gensim.models import KeyedVectors
import time
import tqdm

def compute_dot(caps, imgs, model):
	s = np.squeeze(model.predict([imgs, caps]))
	return s
#
# def construct_matrix(caps, imgs, l, model):
# 	scores = []
# 	mat = np.empty((l,l))
#
# 	print("Constructing Matrix")
# 	print()
# 	for i in tqdm.tqdm(range(l)):
# 		if i == 0:
# 			new_caps = caps
# 		else:
# 			new_caps = np.roll(caps,i,axis=0)
# 		scores.append(compute_dot(new_caps, imgs, model))
#
# 	print("Indexing")
# 	for j in tqdm.tqdm(range(l)):
# 		s = scores[j]
#
# 		for k in range(l):
# 			mat[(j+k)%3,k] = s[k]
#
# 	return mat

def construct_list(caps, imgs, l):
	for i in tqdm.tqdm(range(1,l)):

		new_caps = np.roll(caps,i,axis=0)
		np.concatenate((caps, new_caps))
		np.concatenate((imgs,imgs))
	return caps,imgs



# def process_img_id(imgs, ids):
# 	''' Remove duplicate based on imgs'''
# 	imgid = []
# 	for i in range(len(imgs)):
# 		imgs[i].reshape()
# 		imgid.append([imgs, ids])
#
# 	return ids, imgs


#print(model.summary())
# exit()
word_model = KeyedVectors.load_word2vec_format('word_model.bin')
ids, imgs, caps = get_test("caption_test.json", "../datasets/CUHK-PEDES", word_model)
# ids = ids[:100]
# imgs = imgs[:100]
# caps = caps[:100]
# ids, imgs = process_img_id(imgs, ids)
# print(ids.shape)
# print(caps.shape)
# print(imgs.shape)
# exit()

model = load_model("../default.h5").layers[4]
# s = np.squeeze(model.predict([imgs, caps]))
# print(s[:20])
# exit()
print("data and model loaded")
# ids_unique = np.unique(ids)
# imgs_unique = np.unique(imgs)
l = len(ids)
print("List Constructing")
caps_list,imgs_list = construct_list(caps, imgs, l)

print("Computing")
s = compute_dot(caps_list, imgs_list, model)
s.shape = (l,l)

mat = np.empty((l,l))
mat[0,] = s[0]
for i in range(1,l):
	mat[i,] = np.roll(s[i], -i, axis=0)



rank1 = 0
rank5 = 0
rank10 = 0
rank20= 0
idx = 0
for i in range(l):
	score = mat[:,i]
	idx_score = np.argsort(score)[::-1]
	#end=time.time()-start
	#print("Sorting run in: " + str(end))
	res_ids = ids[idx_score]
	res_img = imgs[idx_score]	#predicted images True image: imgs[cap_id]

	print("True ID: " + str(ids[idx]))
	print("Predicted: ")
	print(res_ids[:40])
	print("")
	match = False
	for i, res_id in enumerate(res_ids):
		if res_id == ids[idx]:
			match = True
		if i == 0 and match:
			rank1 += 1
		if i == 9 and match:
			rank5 += 1
		if i== 19 and match:
			rank10 += 1
		if i == 39:
			if match > 0:
				rank20 += 1

			print("Caption " + str(idx+1))
			print("Rank1: " + str(rank1/(idx+1)))
			print("Rank5: " + str(rank5/(idx+1)))
			print("Rank10: " + str(rank10/(idx+1)))
			print("Rank20: " + str(rank20/(idx+1)))
			print("")
			break
	idx += 1

# rank1 =  rank1/l
# rank5 = rank5/l
# rank10 = rank10/l
# rank20 = rank20/l
#
# print(rank1,rank5,rank10,rank20)
