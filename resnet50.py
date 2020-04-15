from keras.applications.resnet50 import ResNet50
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)
#hyper
pooling = "avg"  #avg, max or none, see https://keras.io/applications/

def run_resnet50(imgs):
	model = ResNet50(weights='imagenet', pooling = pooling)
	features = model.predict(imgs)
	return features
