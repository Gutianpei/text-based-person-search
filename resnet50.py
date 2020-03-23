from keras.applications.resnet50 import ResNet50

#hyper
pooling = "avg"  #avg, max or none, see https://keras.io/applications/

def run_resnet50(imgs):
	model = ResNet50(weights='imagenet', pooling = pooling)
	features = model.predict(imgs)
	return features