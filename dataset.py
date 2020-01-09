# coding=utf-8
"""
@File   : dataset.py
@Time   : 2020/01/07
@Author : Zengrui Zhao
"""
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as tf
import numpy as np
from pathlib import Path
import os
from PIL import Image
import random
import torchvision.transforms.functional as F
from albumentations.augmentations.transforms import RandomCrop
import torch
import matplotlib.pyplot as plt

def getGradient(input, show=False):
    def getSobelKernel(size):
        hvRange = np.arange(-size // 2 + 1, size // 2 + 1, dtype=np.float32)
        h, v = np.meshgrid(hvRange, hvRange)
        kernelH = h / (h * h + v * v + 1.0e-15)
        kernelV = v / (h * h + v * v + 1.0e-15)

        return kernelH, kernelV

    kernelSize = 5
    mh, mv = getSobelKernel(kernelSize)
    mh = np.reshape(mh, [1, 1, kernelSize, kernelSize])
    mv = np.reshape(mv, [1, 1, kernelSize, kernelSize])
    if type(input) is torch.Tensor:
        input = input.cpu().detach().numpy()
        h, v = input[:, 1, ...][:, None, ...], input[:, 0, ...][:, None, ...]
    else:
        h, v = input[1, ...][None, None, ...], input[0, ...][None, None, ...]

    dh = tf.conv2d(torch.tensor(h, dtype=torch.double), torch.tensor(mh, dtype=torch.double), stride=1, padding=2)
    dv = tf.conv2d(torch.tensor(v, dtype=torch.double), torch.tensor(mv, dtype=torch.double), stride=1, padding=2)
    if show:
        size = 200
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].imshow(-np.squeeze(np.array(dh))[0:size, 0:size], cmap='jet')
        axes[0, 1].imshow(np.squeeze(np.array(h))[0:size, 0:size], cmap='jet')
        axes[1, 0].imshow(-np.squeeze(np.array(dv))[0:size, 0:size], cmap='jet')
        axes[1, 1].imshow(np.squeeze(np.array(v))[0:size, 0:size], cmap='jet')
        axes[0, 0].set_title('Gradient', fontsize=20)
        axes[0, 0].set_ylabel('Horizontal', fontsize=20)
        axes[0, 1].set_title('Raw', fontsize=20)
        axes[1, 0].set_ylabel('Vertical', fontsize=20)
        plt.show()
        return
    else:
        return torch.cat((dv, dh), dim=1)


class Data(Dataset):
    def __init__(self, root=Path(__file__),
                 cropSize=(512, 512),
                 isAugmentation=False,
                 mode='train'):
        self.mode = mode
        self.isAugmentation = isAugmentation
        self.crop = True
        self.cropSize = cropSize
        self.root = root
        self.imgs = os.listdir(Path(root) / 'Images')
        assert(mode in ['train', 'test'])
        self.toTensor = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.80508233, 0.80461432, 0.8043749),
                                                                 (0.14636562, 0.1467832,  0.14712358))])

    def augmentation(self, img, mask, horizontal, vertical):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            horizontal = horizontal.transpose(Image.FLIP_LEFT_RIGHT)
            vertical = vertical.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            horizontal = horizontal.transpose(Image.FLIP_TOP_BOTTOM)
            vertical = vertical.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            angle_list = [0, 90, -90, 180]
            angle = angle_list[random.randint(0, len(angle_list) - 1)]
            img = F.rotate(img, angle=angle)
            mask = F.rotate(mask, angle=angle)
            horizontal = F.rotate(horizontal, angle=angle)
            vertical = F.rotate(vertical, angle=angle)
        if random.random() < 0.3:  # brightness
            bf_list = np.linspace(0.8, 1.2, 9)
            bf = bf_list[random.randint(0, len(bf_list) - 1)]
            img = F.adjust_brightness(img, brightness_factor=bf)
        if random.random() < 0.3:  # contrast
            cf_list = np.linspace(0.8, 1.2, 5)
            cf = cf_list[random.randint(0, len(cf_list) - 1)]
            img = F.adjust_contrast(img, contrast_factor=cf)
        if random.random() < 0.3:  # gamma
            gm_list = np.linspace(0.8, 1.2, 5)
            gm = gm_list[random.randint(0, len(gm_list) - 1)]
            img = F.adjust_gamma(img, gamma=gm)
        if random.random() < 0.3:
            hf_list = np.linspace(-0.1, 0.1, 11)
            hf = hf_list[random.randint(0, len(hf_list) - 1)]
            img = F.adjust_hue(img, hue_factor=hf)
        if random.random() < 0.3:
            sf_list = np.linspace(0.8, 1.2, 5)
            sf = sf_list[random.randint(0, len(sf_list) - 1)]
            img = F.adjust_saturation(img, saturation_factor=sf)
        if self.crop:
            transform = RandomCrop(self.cropSize[0], self.cropSize[1])
            wh = transform.get_params()
            img = transform.apply(np.array(img), h_start=wh['h_start'], w_start=wh['w_start'])
            mask = transform.apply(np.array(mask), h_start=wh['h_start'], w_start=wh['w_start'])
            vertical = transform.apply(np.array(vertical), h_start=wh['h_start'], w_start=wh['w_start'])
            horizontal = transform.apply(np.array(horizontal), h_start=wh['h_start'], w_start=wh['w_start'])

        return img, mask, horizontal, vertical

    def __getitem__(self, item):
        imgPath = Path(Path(self.root) / 'Images' / self.imgs[item])
        maskPath = Path(Path(self.root) / 'Labels' / (self.imgs[item].split('.')[0] + '.npy'))
        verticalPath = Path(
            Path(self.root) / 'HorizontalVerticalMap' / (self.imgs[item].split('.')[0] + '_vertical.npy'))
        horizontalPath = Path(
            Path(self.root) / 'HorizontalVerticalMap' / (self.imgs[item].split('.')[0] + '_horizontal.npy'))
        if self.mode == 'train':
            img = Image.open(imgPath).convert('RGB')
            mask = np.load(maskPath)[..., -1]
            mask[mask > 0] = 1
            vertical = np.load(verticalPath)
            horizontal = np.load(horizontalPath)
        else:
            img = Image.open(imgPath).convert('RGB').resize((992, 992))
            mask = np.resize(np.load(maskPath)[..., -1], (992, 992))
            mask[mask > 0] = 1
            vertical = Image.fromarray(np.load(verticalPath)).resize((992, 992))
            horizontal = Image.fromarray(np.load(horizontalPath)).resize((992, 992))

        if self.isAugmentation and self.mode == 'train':
            img, mask, horizontal, vertical = self.augmentation(img,
                                                                Image.fromarray(mask),
                                                                Image.fromarray(horizontal),
                                                                Image.fromarray(vertical))

        horizontal = np.array(horizontal)[..., None]
        vertical = np.array(vertical)[..., None]
        assert(len(np.unique(mask)) == 2)
        horizontalVertical = np.concatenate((vertical, horizontal), axis=-1)

        return self.toTensor(img), mask, np.transpose(horizontalVertical, (2, 0, 1))

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    data = Data(root=Path(__file__).parent.parent / 'data/test', isAugmentation=True, mode='test')
    hv = data[0][2]
    getGradient(hv, show=True)
    # print(np.unique(data[0][2]))
