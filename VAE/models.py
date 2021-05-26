import numpy as np
import torch
import torch.utils.data as torch_data
from torchvision.utils import save_image
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import random
from torch.nn import functional as F

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)
    
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)
    
class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()
        
        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)

class VAE_upsample(nn.Module):
    def __init__(self):
        super(VAE_upsample, self).__init__()   
        base = 16
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=2, padding=1),
            Conv(base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            nn.Conv2d(4*base, 64*base, 8),
            nn.LeakyReLU()
        )
        self.encoder_mu = nn.Conv2d(64*base, 32*base, 1)
        self.encoder_logvar = nn.Conv2d(64*base, 32*base, 1)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(32*base, 64*base, 1),
            ConvTranspose(64*base, 4*base, 8),
            Conv(4*base, 4*base, 3, padding=1),
            ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 2*base, 3, padding=1),
            ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, base, 3, padding=1),
            ConvUpsampling(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class VAE_conv(nn.Module):
    def __init__(self):
        super(VAE_conv, self).__init__()   
        base = 16
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=2, padding=1),
            Conv(base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            Conv(2*base, 2*base, 3, stride=2, padding=1),
            Conv(2*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            Conv(4*base, 4*base, 3, stride=2, padding=1),
            nn.Conv2d(4*base, 64*base, 8),
            nn.LeakyReLU()
        )
        self.encoder_mu = nn.Conv2d(64*base, 32*base, 1)
        self.encoder_logvar = nn.Conv2d(64*base, 32*base, 1)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(32*base, 64*base, 1),
            ConvTranspose(64*base, 4*base, 8),
            Conv(4*base, 4*base, 3, padding=1),
            ConvTranspose(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 4*base, 3, padding=1),
            ConvTranspose(4*base, 4*base, 4, stride=2, padding=1),
            Conv(4*base, 2*base, 3, padding=1),
            ConvTranspose(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, 2*base, 3, padding=1),
            ConvTranspose(2*base, 2*base, 4, stride=2, padding=1),
            Conv(2*base, base, 3, padding=1),
            ConvTranspose(base, base, 4, stride=2, padding=1),
            nn.Conv2d(base, 1, 3, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



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