'''training'''
import torchvision
from gensim.models import Word2Vec
import numpy as np
import dataset

# load images and captions
dpath = "PATH TO DATASET"
data = dataset.dataset(dataset_dir = dpath, new_width = 128, new_height = 256)
images, captions = data.get_img_caption()
###### Renset50 ############
model = torchvision.models.resnet50(pretrained=True)
#model = model.cuda()
resnet_output = model(images)
print(resnet_output.shape)

# word embedding training (could change to pre-trained one)
word_model = Word2Vec(captions, size=300, min_count=1)
word_model.wv.save_word2vec_format('word_model.bin')
# convert (not finished, depends on the shape of captions)
caption_embeddings = np.array([word_model[i] for i in captions])
