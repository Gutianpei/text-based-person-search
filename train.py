'''training'''
import torchvision
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
