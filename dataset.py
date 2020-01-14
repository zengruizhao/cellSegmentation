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
from albumentations.augmentations.transforms import RandomCrop, GaussianBlur, MedianBlur, VerticalFlip, HorizontalFlip, Rotate
import torch
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects, watershed
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from metrics import *

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

def proc(mask, hv):
    h, v = hv[1, ...], hv[0, ...]
    h = cv2.normalize(h, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v = cv2.normalize(v, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    hSobel = cv2.Sobel(h, cv2.CV_64F, 1, 0, ksize=21)
    vSobel = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=21)
    hSobel = 1 - cv2.normalize(hSobel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    vSobel = 1 - cv2.normalize(vSobel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    Sm = np.maximum(hSobel, vSobel)
    Sm = Sm - (1 - mask)
    Sm[Sm < 0] = 0
    # Energy Landscape
    E = (1. - Sm) * mask
    E = -cv2.GaussianBlur(E, (3, 3), 0)

    Sm[Sm >= .4] = 1
    Sm[Sm < .4] = 0
    marker = mask - Sm
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)
    pred = watershed(E, marker, mask=mask)

    a = list(np.unique(pred))[1:]
    np.random.shuffle(a)
    temp = np.zeros_like(pred)
    for idx, i in enumerate(list(a)):
        temp[pred==(idx+1)] = a[idx]
    return temp, E

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
        self.toTensor = transforms.Compose([transforms.ToTensor(),])
                                            # transforms.Normalize((0.80508233, 0.80461432, 0.8043749),
                                            #                      (0.14636562, 0.1467832,  0.14712358))])

    def augmentation(self, img, mask, horizontal, vertical):
        blurM = MedianBlur()
        blurG = GaussianBlur()
        if self.crop:
            while True:
                transform = RandomCrop(self.cropSize[0], self.cropSize[1])
                wh = transform.get_params()
                img_ = transform.apply(np.array(img), h_start=wh['h_start'], w_start=wh['w_start'])
                vertical_ = transform.apply(np.array(vertical), h_start=wh['h_start'], w_start=wh['w_start'])
                horizontal_ = transform.apply(np.array(horizontal), h_start=wh['h_start'], w_start=wh['w_start'])
                mask_ = transform.apply(np.array(mask), h_start=wh['h_start'], w_start=wh['w_start'])
                if len(np.unique(mask_)) == 2:
                    img, vertical, horizontal, mask = Image.fromarray(img_), \
                                                      Image.fromarray(vertical_), \
                                                      Image.fromarray(horizontal_), \
                                                      Image.fromarray(mask_)
                    break
        # if random.random() < .33:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        #     horizontal = horizontal.transpose(Image.FLIP_LEFT_RIGHT)
        #     vertical = vertical.transpose(Image.FLIP_LEFT_RIGHT)
        # elif random.random() < .66:
        #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
        #     mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        #     horizontal = horizontal.transpose(Image.FLIP_TOP_BOTTOM)
        #     vertical = vertical.transpose(Image.FLIP_TOP_BOTTOM)
        # else:
        #     angle = np.random.choice([0, 90, -90, 180])
        #     img = F.rotate(img, angle=angle, resample=Image.BILINEAR)
        #     mask = F.rotate(mask, angle=angle)
        #     horizontal = F.rotate(horizontal, angle=angle)
        #     vertical = F.rotate(vertical, angle=angle)

        if random.random() < 0.3:  # brightness
            bf_list = np.linspace(0.8, 1.2, 9)
            bf = np.random.choice(bf_list)
            img = F.adjust_brightness(img, brightness_factor=bf)
        if random.random() < 0.3:  # contrast
            cf_list = np.linspace(0.8, 1.2, 5)
            cf = np.random.choice(cf_list)
            img = F.adjust_contrast(img, contrast_factor=cf)
        if random.random() < 0.3:  # gamma
            gm_list = np.linspace(0.8, 1.2, 5)
            gm = np.random.choice(gm_list)
            img = F.adjust_gamma(img, gamma=gm)
        if random.random() < 0.3:
            hf_list = np.linspace(-0.1, 0.1, 11)
            hf = np.random.choice(hf_list)
            img = F.adjust_hue(img, hue_factor=hf)
        if random.random() < 0.3:
            sf_list = np.linspace(0.8, 1.2, 5)
            sf = np.random.choice(sf_list)
            img = F.adjust_saturation(img, saturation_factor=sf)
        if random.random() < .5:
            img = blurG.apply(np.array(img))
        else:
            img = blurM.apply(np.array(img))

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
            if self.isAugmentation:
                img, mask, horizontal, vertical = self.augmentation(img,
                                                                    Image.fromarray(mask),
                                                                    Image.fromarray(horizontal),
                                                                    Image.fromarray(vertical))
            horizontal = np.array(horizontal)[..., None]
            vertical = np.array(vertical)[..., None]
            mask = np.array(mask)
            assert (len(np.unique(mask)) == 2)
            horizontalVertical = np.concatenate((vertical, horizontal), axis=-1)
            return self.toTensor(img), mask[None, ...], np.transpose(horizontalVertical, (2, 0, 1))
        else:
            img = np.array(Image.open(imgPath).convert('RGB').resize((992, 992)))
            mask = np.array(Image.fromarray(np.load(maskPath)[..., 0]).resize((992, 992)), dtype=np.int16)
            return self.toTensor(img), mask

    def __len__(self):
        return len(self.imgs)

class Datatemp(Dataset):
    def __init__(self, root=Path(__file__)):
        self.root = root
        self.imgs = os.listdir(Path(root) / 'Images')

    def __getitem__(self, item):
        maskPath = Path(Path(self.root) / 'Labels' / (self.imgs[item].split('.')[0] + '.npy'))
        verticalPath = Path(
            Path(self.root) / 'HorizontalVerticalMap' / (self.imgs[item].split('.')[0] + '_vertical.npy'))
        horizontalPath = Path(
            Path(self.root) / 'HorizontalVerticalMap' / (self.imgs[item].split('.')[0] + '_horizontal.npy'))

        mask = np.array(np.load(maskPath), dtype=np.int16)
        mask0 = mask[..., 0]
        mask1 = mask[..., 1]
        mask1[mask1 > 0] = 1
        vertical = np.load(verticalPath)
        horizontal = np.load(horizontalPath)
        horizontal = np.array(horizontal)[..., None]
        vertical = np.array(vertical)[..., None]
        horizontalVertical = np.concatenate((vertical, horizontal), axis=-1)

        return mask0, mask1, np.transpose(horizontalVertical, (2, 0, 1))


    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    data = Datatemp(root=Path(__file__).parent.parent / 'data/train')
    for i in data:
        mask0, mask1, hv = i[0], i[1], i[2]
        pred, E = proc(mask1, hv)
        metricPQ, _ = get_fast_pq(mask0, pred)
        metricAJI = get_fast_aji_plus(mask0, pred)
        metricDice2 = get_fast_dice_2(mask0, pred)
        print(f'AJI: {metricAJI:.4f}, '
              f'Dice2: {metricDice2:.4f}, '
              f'dq: {metricPQ[0]:.4f}, '
              f'sq: {metricPQ[1]:.4f}, '
              f'pq: {metricPQ[2]:.4f}')
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(mask0, cmap='jet')
        ax[1].imshow(pred, cmap='jet')
        ax[2].imshow(E, cmap='jet')
        ax[0].set_title('Mask', fontsize=20)
        ax[1].set_title('Proc', fontsize=20)
        ax[2].set_title('Energy', fontsize=20)
        plt.show()
        break
    # data = Data(root=Path(__file__).parent.parent / 'data/train',
    #            mode='train',
    #            isAugmentation=True,
    #            cropSize=(384, 384))
    # for i in data:
    #     img = np.transpose(np.array(i[0]), (1, 2, 0))
    #     mask = np.squeeze(i[1])
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(img)
    #     ax[1].imshow(mask)
    #     plt.show()

    # getGradient(hv, show=True)
    # print(np.unique(data[0][2]))
