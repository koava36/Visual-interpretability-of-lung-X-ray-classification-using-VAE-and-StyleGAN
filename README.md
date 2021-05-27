# Generative-models-for-lung-parenchyma-progression-modeling


## Prerequisites

Google Colab GPU and .
Main prerequisites are:

1. [Pytorch](https://pytorch.org/)
2. [CUDA](https://developer.nvidia.com/cuda-downloads)

## Experiments

- ```./VAE/algorithm_with_VAE_part1.ipynb ``` -- __Variational Autoencoder__ trained on initial images with/without perceptual, l1 and kl losses and implementation of algorithm;


## Pretrained models

Checkpoints for all necessary models: 

["Checkpoints"](https://drive.google.com/drive/u/1/folders/1o8Gr2bwNK_TzF5MxUdKNjj2Wb8RiveL0)

## Data

In this project we use  data from the [“COVID-19 Image Data Collection”](https://arxiv.org/abs/2003.11597), [git](https://github.com/ieee8023/covid-chestxray-dataset) (Cohen et al., 2020) chest X-ray dataset. This dataset was collected from public sources as well as through indirect collection from hospitals and physicians It contains three classes: 1) healthy, 2)community acquired pneumonia (CAP), and 3) COVID-19 

To download preprocessed data: ["Preprocessed data"](https://drive.google.com/drive/u/1/folders/1eWKsLpFsz4F57q4VNZmiBL2ap1e0k6Um)

Data splitted on train/test/val and have corresponding labels for images in ```train.csv/test.csv/val.csv ``` files.

To preprocess images by yourself follow the notebook:
```Data_downloader_and_extraction.ipynb ``` 

## Results

- __Classifier__ 




- __Variational autoencoder__ 

Different variations of losses were considered. For l1 + kl loss ConvTranspose in decoder layer show better results, than upsampling + convolution.
For perceptual loss we use upsampling + conv structure of decoder to decrease the effect of stripes. Also for perceptual loss we use pretrained on train dataset ```DenseNet161``` model.

|losses | structure of decoder | l1 loss | PSNR| SSIM|
|----------------|:---------:|----------------:|--------------:|-----------:|
| Perceptual + l1 + kl| upsampling + conv| 0.075|25.52|0.67|
|l1 + kl| ConvTranspose |0.04 |28.54|0.77|

__Perceptual + l1 + kl losses:__

<img src="https://github.com/koava36/Generative-models-for-lung-parenchyma-progression-modeling/blob/main/imgs/perceptual_vae.png" alt="" width="500" height="300">

__l1 + kl losses:__

<img src="https://github.com/koava36/Generative-models-for-lung-parenchyma-progression-modeling/blob/main/imgs/vae_reconstructed.png" alt="" width="500" height="300">



- __Style Gan2__

For StyleGan2 we took pretrained on XRAY images [checkpoint](https://drive.google.com/file/d/1Wmi7qZ-ngX0clvHmgS-j2-FNIMZCWziK/view) and then finetune it on our data.

<img src="https://github.com/koava36/Generative-models-for-lung-parenchyma-progression-modeling/blob/main/imgs/style_gan_generated.png" alt="" width="1000" height="200">



<!-- 
<!-- Build LR classifier both on initial data and obtained after encoder bottleneck vectors for gender and age labels.

| Label| Initial vectors | Bottlenck vectors |
|----------------|:---------:|----------------:|
| Gender | 0.919 | 0.856 |
| Age| 0.47 | 0.465 | -->
 




