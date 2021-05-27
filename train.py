# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Modifications Copyright Skoltech Deep Learning Course.

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from scipy import stats

from model import Generator, Discriminator
from loss import GANLoss


class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-0.01,0.01)


class GAN(pl.LightningModule):
    def __init__(
        self,
        loss_type: str,
        class_conditional: bool,
        truncation_trick: bool,
        train_dataset,
        batch_size: int = 128,
        lr_gen: float = 1e-5, #1e-4
        lr_dis: float = 8e-6 #4e-4
    ):
        super(GAN, self).__init__()
        self.noise_channels = 512
        self.class_conditional = class_conditional
        self.truncation_trick = truncation_trick
        self.loss_type = loss_type

        self.batch_size = batch_size
        
        if loss_type == 'non_saturating':
            self.lr_gen: float = 6e-4
            self.lr_dis: float = 7e-4
        elif loss_type == 'hinge':
            self.lr_gen: float = 1e-5
            self.lr_dis: float = 8e-6
	elif loss_type == 'wasserstein':
	    self.lr_gen: float = 6e-4
            self.lr_dis: float = 7e-4

        self.eps = 1e-7
        self.num_classes = 3

        self.train_dataset = train_dataset

        self.gen = Generator(
            min_channels=8, 
            max_channels=128, 
            noise_channels=self.noise_channels, 
            num_classes=self.num_classes, 
            num_blocks=4,
            use_class_condition=class_conditional)

        self.dis = Discriminator(
            min_channels=8, 
            max_channels=128, 
            num_classes=self.num_classes, 
            num_blocks=4,
            use_projection_head=class_conditional)

	self.clipper = WeightClipper()

        self.loss = GANLoss(loss_type)

        self.register_buffer('val_noise', torch.randn(len(self.train_dataset), self.noise_channels))

        if self.truncation_trick:
        	# TODO: replace val_noise.data with truncated normal samples 1 (point)
        	# For that, you can use SciPy.stats library
        	
            self.register_buffer('val_noise', torch.tensor(stats.truncnorm.rvs(-1, 1, size=(len(self.train_dataset), self.noise_channels))))

    def forward(self, noise, labels):
        return self.gen(noise, labels)

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, labels = batch
        
        # TODO: (1 point)
        #   - Sample input noise 
        #   - Forward pass and calculate loss
        # 	  If optimizer_idx == 0: calc gen loss
        # 	  If optimizer_idx == 1: calc dis loss

	
        noise = torch.randn(len(imgs), self.noise_channels, device='cuda:0')
        
        fake_imgs = self.forward(noise, labels)
        real_scores = self.dis(imgs, labels)
        fake_scores = self.dis(fake_imgs, labels)
        
        if self.loss_type == 'non_saturating':
            real_scores = torch.sigmoid(real_scores)
            fake_scores = torch.sigmoid(fake_scores)
	if self.loss_type == 'wasserstein':
            real_scores = nn.ReLU(real_scores)
            fake_scores = nn.ReLU(fake_scores)
        
        print('fake: ', torch.mean(fake_scores))
        print('real: ', torch.mean(real_scores))
        
        if optimizer_idx == 0:
            loss = self.loss(fake_scores)
        elif optimizer_idx == 1:
            loss = self.loss(fake_scores, real_scores)
	    if self.loss_type == 'wasserstein':
	    	self.dis.apply(self.clipper)

        assert loss.shape == (), 'Loss should be a single digit'

        loss_name = 'loss_gen' if optimizer_idx == 0 else 'loss_dis'
        self.log(loss_name, loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
	if self.loss_type == 'wasserstein':
	   opt_gen = torch.optim.RMSprop(self.gen.parameters(), lr=self.lr_gen)
           opt_dis = torch.optim.RMSprop(self.dis.parameters(), lr=self.lr_dis)
	else:
           opt_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_gen, betas=(0.0, 0.999))
           opt_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_dis, betas=(0.0, 0.999))

        return [opt_gen, opt_dis], []

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=2, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=2, batch_size=self.batch_size, shuffle=False)

    @torch.no_grad()
    def on_epoch_end(self):
        # Visualize Images
        noise = self.val_noise[:self.num_classes * 4]
        labels = torch.arange(self.num_classes).repeat_interleave(4, dim=0).to(noise.device)

        fake_imgs = self.forward(noise, labels)
        fake_imgs = fake_imgs.detach().cpu()
        

        grid = utils.make_grid(fake_imgs, nrow=4)
        self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
