'''training'''
import torchvision
import dataset

dpath = "PATH TO DATASET"
images, captions = dataset.dataset(dataset_dir = dpath)
###### Renset50 ############
model = torchvision.models.resnet50(pretrained=True)
#model = model.cuda()
resnet_output = model(images)
print(resnet_output.shape)
