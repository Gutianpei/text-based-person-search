from keras.models import load_model
from keras.models import Model

model = load_model("../default.h5").layers[4]

img_input = model.layers[0].input
img_output = model.layers[-3].output
img_model = Model(img_input, img_output)

cap_input = model.layers[1].input
cap_output = model.layers[-2].input
cap_model = Model(cap_input, cap_output)
print(img_model.summary())
    #
    # resnet = ResNet50(include_top = False, weights = 'imagenet', input_shape=(384,128,3))
    # output = resnet.layers[-1].output
    # #output = Dense(1024)
    # resnet = Model(resnet.input, output=output)
    # for layer in resnet.layers:
    #     layer.trainable = False
