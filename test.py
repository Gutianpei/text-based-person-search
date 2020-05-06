'''Script for testing
'''
from keras.models import load_model
import numpy as np
from testin_data import get_test
from collections import Counter


def compute_dot(cap, imgs, model):
	score = []
	for img in imgs:
		score.append(model.predict([cap, img]))
	return score

model = load_model("").layers[4]

ids, imgs, caps = get_test("caption_test.json","../datasets/CUHK-PEDES")

# ids_unique = np.unique(ids)
# imgs_unique = np.unique(imgs)

rank1, rank5, rank10, rank20 = 0

for cap in caps:
	score = compute_dot(cap, imgs, model)
	idx_score = np.argsort(score)
	res_ids = ids[idx_score]

	for i, res_id in enumerate(res_ids):
		match = 0
		if res_id == ids[idx]:
			match += 1
		if i == 0 and match > 0:
			rank1 += 1
		if i == 4 and match > 0:
			rank5 += 1
		if i==9 and match > 0:
			rank10 += 1

		if i == 19:
			if match > 0:
				rank20 += 1
			break


rank1 /=  len(caps)
rank5 /= len(caps)
rank10 /= len(caps)
rank20 /= len(caps)

print(rank1,rank5,rank10,rank20)
