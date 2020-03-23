'''training'''

from gensim.models import Word2Vec
import numpy as np
import tensorflow as tf
from keras.preprocessing import sequence
import dataset
from scipy import spatial
from resnet50 import run_resnet50
import keras.backend as K
from keras.models import model_from_yaml
import os
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional

def euclidean_distance_loss(y_true, y_pred):
	''' l2 norm loss function
	
		Args: 
			y_pred: bi-lstm output vector
			y_true: resnet50 output vector
		Return:
			keras backend loss function
	'''
	return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def cos_distance_loss(y_true, y_pred):
	''' cos distance loss function'''
	y_true = K.l2_normalize(y_true, axis=-1)
	y_pred = K.l2_normalize(y_pred, axis=-1)
	return K.mean(1 - K.sum((y_true * y_pred), axis=-1))
	
	
def main():
	# Hyper params and configs #
	use_cuda = False 
	lr = 0.00015
	batch_size = 128
	model_path = "models"
	dpath = "PATH TO DATASET" #dataset path
	ratio = 0.6	 #train/test split ratio
	max_features = 20000
	maxlen = 100
	activition = "relu"
	############################
	#setup gpu
	if use_cuda = True:
		config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
		sess = tf.Session(config=config) 
		keras.backend.set_session(sess)
	
	curr_path = os.getcwd()
	# Load images and captions #
	data = dataset.dataset(dataset_dir = dpath, new_width = 128, new_height = 256, ratio = 0.6)
	images, captions = data.get_img_caption()
	############################

	###### Renset50 ############ 
	resnet_feat = run_resnet50(imgs = images)
	print("Feature extraced by resnet50, feature vector shape: " + str(resnet_feat.shape))
	print("#"*28 + "\n")
	############################

	###### Bi-LSTM #############
	print("Building Bi-LSTM")
	#tip: dataset.py里面提取caption还没写，我不知道要把caption变成什么格式，下面这些能不能加到dataset.py里？因为test的时候还要再用一次

	# word embedding training (could change to pre-trained one)
	word_model = Word2Vec(captions, size=300, min_count=1)
	word_model.wv.save_word2vec_format('word_model.bin')

	# convert
	captions_embeddings = np.array([word_model[i] for i in captions])
	captions_embeddings = sequence.pad_sequences(captions_embeddings, dtype='float32', padding='post')
	
	model = Sequential()
	model.add(Embedding(max_features, 128, input_length=maxlen))
	model.add(Bidirectional(LSTM(64)))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation=activition))
	
	model.compile("adam",loss=cos_distance_loss,metrics=['accuracy']) #or loss=euclidean_distance_loss 

#	model.fit(x_train, resnet_feat,
#						batch_size=batch_size,
#						epochs=100,
#						validation_data=[x_test, y_test])
	############################

	###### save model ##########
	mpath = os.join(curr_path,model_path)
	print("Saving model to " + mpath)
	
	model_yaml = model.to_yaml()
	with open(mpath + "model.yaml", "w") as yaml_file:
		yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to " + mpath + "model.yaml")
	
	
if __name__ == "__main__":
	main()

