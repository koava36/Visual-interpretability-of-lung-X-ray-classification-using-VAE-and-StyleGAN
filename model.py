import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm



class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        self.batchnorm = nn.BatchNorm2d(num_features, affine=False)
        self.linear1 = nn.Linear(embed_features, num_features)
        self.linear2 = nn.Linear(embed_features, num_features)

    def forward(self, inputs, embeds):
        gamma = self.linear1(embeds.float())
        bias = self.linear2(embeds.float())
    
        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = self.batchnorm(inputs)

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!
        
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_channels = embed_channels
        self.batchnorm = batchnorm
        self.do_upsample = upsample
        self.do_downsample = downsample
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        
        if batchnorm:
            self.batchnorm1 = AdaptiveBatchNorm(in_channels, embed_channels)
            self.batchnorm2 = AdaptiveBatchNorm(out_channels, embed_channels)
            
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.downsample = nn.AvgPool2d(kernel_size=2)
        self.skip_con_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        
    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        
        if self.do_upsample:
            inputs = self.upsample(inputs)
        
        if not self.batchnorm:
            x = self.act1(inputs).type_as(inputs)
            x = self.conv1(x)
            x = self.act2(x)
            x = self.conv2(x)
            
        else:
            
            x = self.batchnorm1(inputs, embeds).type_as(inputs)
            x = self.act1(x)
            x = self.conv1(x)
            x = self.batchnorm2(x, embeds)
            x = self.act2(x)
            x = self.conv2(x)
        
        if self.in_channels != self.out_channels:
            inputs = self.skip_con_conv(inputs)
            
        outputs = inputs + x
        
        if self.do_downsample:
            outputs = self.downsample(outputs)
        
        
        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.init_size = 8
        self.output_size = self.init_size * 2**num_blocks
        self.max_channels = max_channels
        self.noise_channels = noise_channels
        self.use_class_condition = use_class_condition
        
        if use_class_condition:
            self.embedding = nn.Embedding(num_classes, noise_channels)
            self.linear = spectral_norm(nn.Linear(2 * noise_channels, max_channels * self.init_size * self.init_size))
            embed_channels = 2 * noise_channels
        else:
            self.linear = spectral_norm(nn.Linear(noise_channels, max_channels * self.init_size * self.init_size))
            embed_channels = noise_channels
        
        self.resblocks = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = max_channels // (2 ** i)
            out_channels = max_channels // (2 ** (i + 1))
            self.resblocks.append(PreActResBlock(in_channels, out_channels, embed_channels, batchnorm=True, upsample=True))
         
        
        self.final_conv = nn.Sequential(nn.BatchNorm2d(out_channels),
                                        nn.ReLU(),
                                        spectral_norm(nn.Conv2d(min_channels, 1, kernel_size=3, padding=1)))

    def forward(self, noise, labels=None):
        
        embed = noise
        
        if self.use_class_condition:
            class_embed = self.embedding(labels)
            embed = torch.cat((noise, class_embed), 1)
        
        x = self.linear(embed.float())
        x = x.reshape(-1, self.max_channels, self.init_size, self.init_size)
        
        for layer in self.resblocks:
            x = layer(x, embed)
            
        x = self.final_conv(x)
        outputs = torch.sigmoid(x)
        
        
        assert outputs.shape == (noise.shape[0], 1, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        
        self.use_projection_head = use_projection_head
        self.max_channels = max_channels
        
        self.embedding = spectral_norm(nn.Embedding(num_classes, max_channels))
        self.fitst_conv = nn.Sequential(spectral_norm(nn.Conv2d(1, min_channels, kernel_size=3, padding=1)),
                                        nn.ReLU())
        self.resblocks = nn.ModuleList()
        for i in range(num_blocks):
            in_channels = min_channels * (2 ** i)
            out_channels = min_channels * (2 ** (i + 1))
            self.resblocks.append(PreActResBlock(in_channels, out_channels, batchnorm=False, downsample=True))
        self.act = nn.ReLU()
        self.pooling = nn.LPPool2d(1, (8, 8))
    
        self.linear = spectral_norm(nn.Linear(max_channels, 1))

    def forward(self, inputs, labels):
        
        x = self.fitst_conv(inputs)
        
        for conv_block in self.resblocks:
            x = conv_block(x)
        
        x = self.pooling(self.act(x))
        x = x.reshape(-1, self.max_channels)
        
        
        if self.use_projection_head:
            y = self.embedding(labels)
            
            inner_prod = torch.diag(x @ y.T).reshape(-1, 1)
            x = self.linear(x)
            
            scores = x + inner_prod
            scores = scores.reshape(-1, )
            
        else:
            scores = self.linear(x).reshape(-1, )

        assert scores.shape == (inputs.shape[0],)
        return scores
