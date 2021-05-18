import os
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import torch
from nibabel.testing import data_path
import nibabel
from tqdm.notebook import tqdm

class CT_Covid(Dataset):
    def __init__(self, path='./train'):
        # load all nii handle in a list
        self.labels = []
        files_list = []
        for root, dirs, files in os.walk(path):
          for i in tqdm(range(len(files))):
            files_list.append(root + '/' + files[i])
            self.labels.append(int(root.split('-')[-1]))
        self.images_list = [nib.load(img) for img in files_list]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        nii_image = self.images_list[idx]
        data = torch.from_numpy(np.asarray(nii_image.dataobj))
        target = self.labels[idx]
        return data, target

def collate_fn(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pad_size = max([i[0].shape[2] for i in data])
    labs = [i[1] for i in data]
    bs = len(data)
    h, w = 512, 512
    image = torch.zeros((bs, h, w, pad_size), dtype=torch.long)
    labels = torch.tensor(labs, dtype=torch.long)
    for i in range(len(data)):
      padded_len = pad_size - data[i][0].shape[2]
      image[i,...] = torch.nn.functional.pad(data[i][0], (0, padded_len))

    image=image[..., image.shape[3] // 2]
      
    return image.to(device), labels.to(device)
