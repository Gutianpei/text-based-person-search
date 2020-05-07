'''Script for testing
'''
from keras.models import load_model
import numpy as np
from testing_data import get_test
from collections import Counter
from gensim.models import KeyedVectors
import time
import tqdm

def compute_dot(cap, imgs, model):
	score = []
	cap.shape = (1,50,50)
	for img in tqdm.tqdm(imgs):
		img.shape = (1,384,128,3)
		#start = time.time()
		score.append(model.predict([img, cap]))
		#end=time.time()-start
		#print("Model run in: " + str(end))
	return score

def process_img_id(imgs, ids):
	''' Remove duplicate based on imgs'''
	imgid = []
	for i in range(len(imgs)):
		imgs[i].reshape()
		imgid.append([imgs, ids])

	return ids, imgs


#print(model.summary())
# exit()
word_model = KeyedVectors.load_word2vec_format('word_model.bin')
ids, imgs, caps = get_test("caption_test.json", "../datasets/CUHK-PEDES", word_model)
ids, imgs = process_img_id(imgs, ids)
# print(ids.shape)
# print(caps.shape)
# print(imgs.shape)
# exit()

model = load_model("default.h5").layers[4]

print("data and model loaded")
# ids_unique = np.unique(ids)
# imgs_unique = np.unique(imgs)

rank1 = 0
rank5 = 0
rank10 = 0
rank20= 0

for cap_id, cap in enumerate(caps):
	s = start = time.time()
	score = compute_dot(cap, imgs, model)
	e=time.time()-s
	print("---------------------------------")
	print("Total model run in: " + str(e))

	#start = time.time()
	idx_score = np.argsort(score)[::-1]
	#end=time.time()-start
	#print("Sorting run in: " + str(end))
	res_ids = ids[idx_score]
	res_img = imgs[idx_score]	#predicted images True image: imgs[cap_id]

	print("True ID: " + str(ids[cap_id]))
	print("Predicted: ")
	print(res_ids[:20])
	print("")
	for i, res_id in enumerate(res_ids):
		match = False
		if res_id == ids[cap_id]:
			match = True
		if i == 0 and match:
			rank1 += 1
		if i == 4 and match:
			rank5 += 1
		if i==9 and match:
			rank10 += 1
		if i == 19:
			if match > 0:
				rank20 += 1

			print("Caption " + str(cap_id+1))
			print("Rank1: " + str(rank1/(cap_id+1)))
			print("Rank5: " + str(rank5/(cap_id+1)))
			print("Rank10: " + str(rank10/(cap_id+1)))
			print("Rank20: " + str(rank20/(cap_id+1)))
			print("")
			break


rank1 =  rank1/len(caps)
rank5 = rank5/len(caps)
rank10 = rank10/len(caps)
rank20 = rank20/len(caps)

print(rank1,rank5,rank10,rank20)
