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
Data splitted on train/test/val and have corresponding labels for images in train.csv/test.csv/val.csv files.

To preprocess images by yourself:
```Data_downloader_and_extraction.ipynb ``` -- __Steps for downloading__ 

