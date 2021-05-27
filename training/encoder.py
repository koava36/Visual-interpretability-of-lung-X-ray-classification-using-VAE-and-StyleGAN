import numpy as np
import torch
import torch.utils.data as torch_data
from torchvision.utils import save_image
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import random
from torch.nn import functional as F


class ResNet50_encoder(nn.Module):
    def __init__(self):
      super(ResNet50_encoder, self).__init__()   
      main =  torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=False)
      main.conv1 = nn.Conv2d(1, 64, 7, 2, 3)

      head = list(main.children())[:-5]

      last =  [nn.Conv2d(256, 512, 1, 2),
                    nn.BatchNorm2d(512),
                    nn.ReLU(), 
                    nn.Conv2d(512, 512, 1, 2),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, 1, 2),
                    nn.BatchNorm2d(512),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, 1, 2, 1)]
      self.model = nn.Sequential(*(head + last))
      
    def forward(self, x):
      return self.model(x)