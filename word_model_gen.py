import json
from gensim.models import Word2Vec
import numpy as np

vector_size = 50

js_data = json.load(open(self.dataset_dir + "/caption_all.json"))
captions = [i['captions'] for i in js_data]
flatten_caps = sum(captions, [])

# tokenize
tokenizer = RegexpTokenizer(r'\w+')
tokens = [[j.lower() for j in tokenizer.tokenize(i)] for i in flatten_caps]

# word embedding training (could change to pre-trained one)
word_model = Word2Vec(tokens, size=vector_size, min_count=1)
word_model.wv.save_word2vec_format('word_model.bin')
