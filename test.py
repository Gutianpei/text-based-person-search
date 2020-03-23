'''Script for testing
'''
from keras.models import model_from_yaml
import numpy
import os
import dataset

def main():
	yaml_file = open('model.yaml', 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	loaded_model = model_from_yaml(loaded_model_yaml)
	# load weights into new model
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")
	 
	# evaluate loaded model on test data
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	score = loaded_model.evaluate(X, Y, verbose=0)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
	
if __name__ == "__main__":
	main()
	
