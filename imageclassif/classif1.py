# https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
from torchvision import models
import torch

alexnet = models.alexnet(pretrained=True)

print(alexnet)

