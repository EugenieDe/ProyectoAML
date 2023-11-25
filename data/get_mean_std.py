import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor,Compose
import math
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Create Custon DataLoader   --------------------------------------------------------------------------------

class CustomDataset(Dataset):
    def __init__(self, name, root, mode=None, size=None, id_path=None, nsample=None, transform=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.transform = transform

        with open('splits/%s/train.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        img = self.transform(img)
        mask = self.transform(mask)
        print(img)
        print(mask)
        return img, mask
    


data_transform = Compose([
    ToTensor(),         # Convert images to tensors   
])


# Assuming you have 'train', 'test', and 'validation' folders in your current directory
train_dataset = CustomDataset('endovis2018', "/home/eugenie/These/UniMatch/splits/endovis2018/",transform=data_transform)

# Create DataLoader instances
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=6)

# Get mean and std
mean = 0.
std = 0.
for images, _ in train_loader:
    batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)

mean /= len(train_loader.dataset)
std /= len(train_loader.dataset)
torch.set_printoptions(precision=10)
print("-------------------------------------------------")
print(mean,std)