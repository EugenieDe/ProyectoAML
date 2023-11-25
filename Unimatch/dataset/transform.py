import random

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
from torchvision import transforms
import torchvision.transforms.functional as F


def transform(img, mask=None):
    if mask is not None:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
        ])
        img = trans(img)
        mask = trans(mask)
        r = transforms.RandomRotation.get_params((-90,90))
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(128,128))
        img = F.hflip(img)
        mask = F.hflip(mask)
        if random.random() > 0.5:
            img = F.rotate(img,r)
            mask = F.rotate(mask,r)
        if random.random() > 0.5:
            img = F.vflip(img)
            mask = F.vflip(mask)

        img = transforms.ToPILImage()(img)
        mask = transforms.ToPILImage()(mask)
        return img, mask
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.RandomApply([transforms.RandomRotation(90)], p=0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(1.0),
            transforms.ToPILImage(),
        ])
        return transform()(img)


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)

    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.464, 0.341, 0.363], [0.201, 0.185, 0.196]),
    ])(img) 
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img

def resize(img, mask=None):
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256,256)),
        transforms.ToPILImage(),
    ])
    img = trans(img)
    if mask is not None:
        mask = trans(mask)
        return img, mask
    return img

def random_rot(img, mask=None):
    transform = transforms.Compose([
    transforms.RandomRotation(90),
    ])
    if mask is not None:
        img, mask = transform(img, mask)
        return img, mask
    else:
        return img

def random_flip(img, mask=None):
    transform = transforms.Compose([
    transforms.RandomVerticalFlip(0.5),
    ])
    if mask is not None:
        img, mask = transform(img, mask)
        return img, mask
    else:
        return img

def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask
