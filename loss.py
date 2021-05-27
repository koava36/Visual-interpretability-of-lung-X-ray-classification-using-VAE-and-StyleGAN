import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
import numpy as np
from scipy import linalg



class GANLoss(nn.Module):
    """
    GAN loss calculator

    Variants:
      - non_saturating
      - hinge
    """
    def __init__(self, loss_type):
        super(GANLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, fake_scores, real_scores=None):
        self.relu = nn.ReLU()
        if real_scores is None:
            # TODO: calculate generator loss (2 points)
            if self.loss_type == 'non_saturating':
                loss = -torch.mean(torch.log(torch.sigmoid(fake_scores) + 1e-9))
            elif self.loss_type == 'hinge':
                loss = -torch.mean(fake_scores)
	    elif self.loss_type == 'wasserstein':
		loss = -torch.mean(fake_scores) 

        else:
            # TODO: calculate discriminator loss (2 points)
            if self.loss_type == 'non_saturating':
                l1 = torch.log(torch.sigmoid(real_scores) + 1e-9) 
                l2 = torch.log(1. - torch.sigmoid(fake_scores) + 1e-9)
                loss = -torch.mean(l1+l2)
            elif self.loss_type == 'hinge':
                loss = torch.mean(self.relu(1.0-real_scores)) + torch.mean(self.relu(1.0+fake_scores))
	    elif self.loss_type == 'wasserstein':
		loss = -torch.mean(fake_scores) + torch.mean(real_scores)

        return loss

class ValLoss(nn.Module):
    """
    Calculates FID and IS
    """
    def __init__(self):
        super(ValLoss, self).__init__() ##fixed to ValLoss
        self.inception_v3 = models.inception_v3(pretrained=True)
        self.inception_v3.eval()

        for p in self.inception_v3.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def _features(self, x: torch.Tensor) -> torch.Tensor:
        # Preprocess data
        x = F.interpolate(x, size=(299, 299), mode='bilinear')
        x = (x - 0.5) * 2

        # N x 3 x 299 x 299
        x = self.inception_v3.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_v3.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_v3.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_v3.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_v3.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_v3.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_v3.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_v3.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_v3.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_v3.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_v3.Mixed_7c(x)
        # Adaptive average pooling
        x = self.inception_v3.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception_v3.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)

        return x

    @torch.no_grad()
    def _classifier(self, x: torch.Tensor) -> torch.Tensor:
        # N x 2048
        x = self.inception_v3.fc(x)
        # N x 1000 (num_classes)
        x = F.softmax(x, dim=1)

        return x

    def calc_data(self, real_inputs: list, fake_inputs: list):
        real_features = []
        for real_inputs_batch in real_inputs:
            real_features_batch = self._features(real_inputs_batch)
            real_features.append(real_features_batch.detach().cpu().numpy())            
        real_features = np.concatenate(real_features)

        fake_features = []
        fake_probs = []

        for fake_inputs_batch in fake_inputs:
            fake_features_batch = self._features(fake_inputs_batch)
            fake_probs_batch = self._classifier(fake_features_batch)

            fake_features.append(fake_features_batch.detach().cpu().numpy())
            fake_probs.append(fake_probs_batch.detach().cpu().numpy())

        fake_features = np.concatenate(fake_features)
        fake_probs = np.concatenate(fake_probs)

        return real_features, fake_features, fake_probs

    @staticmethod
    def calc_fid(real_features, fake_features):

    
      mu_r = np.mean(real_features)
      mu_g = np.mean(fake_features)
      sigma_r = np.cov(real_features, ddof=1)
      sigma_g = np.cov(fake_features, ddof=1)
      #(mu_r-mu_g)@(mu_r-mu_g)
      A = sigma_r @ sigma_g
      sqrtA = linalg.sqrtm(A)
      fid =  (np.linalg.norm(mu_r-mu_g))**2 +np.trace(sigma_r+sigma_g - 2*sqrtA)
      
      return fid   # TODO (2 points)

    @staticmethod
    def calc_is(fake_probs):
        #q = np.log(fake_probs/np.mean(fake_probs))
        #x = np.mean(fake_probs * q)
        eps=1e-9
        p_y = np.expand_dims(fake_probs.mean(axis=0), 0)
        kl_d = fake_probs * (np.log(fake_probs + eps) - np.log(p_y + eps))
        sum_kl_d = kl_d.sum(axis=1)
        x = np.mean(sum_kl_d)

        
        return np.exp(x)

    def forward(self, real_images: list, fake_images: list) -> torch.Tensor:
        real_features, fake_features, fake_probs = self.calc_data(real_images, fake_images)

        fid = self.calc_fid(real_features, fake_features)

        inception_score = self.calc_is(fake_probs)

        return fid, inception_score
