# coding=utf-8
"""
@File   : postProcess.py
@Time   : 2020/01/09
@Author : Zengrui Zhao
"""
import numpy as np
import cv2
from skimage.morphology import remove_small_objects, watershed
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
import matplotlib.pyplot as plt

def proc(pred):
    thresSeg, thresSm = .95, .8
    pred = pred.cpu().detach().numpy()
    seg, h, v = pred[0, ...,], pred[1, ...], pred[2, ...]
    seg[seg >= thresSeg] = 1
    seg[seg < thresSeg] = 0
    seg = measurements.label(seg)[0]
    seg = remove_small_objects(seg, min_size=10)
    seg[seg > 0] = 1
    ##
    h = cv2.normalize(h, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    v = cv2.normalize(v, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    hSobel = cv2.Sobel(h, cv2.CV_64F, 1, 0, ksize=21)
    vSobel = cv2.Sobel(v, cv2.CV_64F, 0, 1, ksize=21)
    hSobel = 1 - cv2.normalize(hSobel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    vSobel = 1 - cv2.normalize(vSobel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    Sm = np.maximum(hSobel, vSobel)
    Sm = Sm - (1 - seg)
    Sm[Sm < 0] = 0
    # Energy Landscape
    E = (1. - Sm) * seg
    E = -cv2.GaussianBlur(E, (3, 3), 0)
    #
    Sm[Sm >= thresSm] = 1
    Sm[Sm < thresSm] = 0
    marker = seg - Sm
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = remove_small_objects(np.array(marker, bool), min_size=10)
    marker = measurements.label(marker)[0]
    pred = watershed(E, marker, mask=seg)
    # shuffle
    a = list(np.unique(pred))[1:]
    np.random.shuffle(a)
    temp = np.zeros_like(pred)
    for idx, i in enumerate(list(a)):
        temp[pred == (idx + 1)] = a[idx]
    pred = temp

    return pred

if __name__ == '__main__':
    pass

