'''training'''
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model = model.cuda()
