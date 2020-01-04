# coding=utf-8
"""
@File   : horizontalVerticalMap.py
@Time   : 2020/01/03
@Author : Zengrui Zhao
"""
import time
import numpy as np
import argparse
from pathlib import Path
import os
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from tqdm import tqdm

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--path', type=str, default='../data/train/Labels')
    parse.add_argument('--savePth', type=str, default='../data/train/HorizontalVerticalMap')

    return parse.parse_args()

def mapGenarating(props, img, mode='v'):
    result = np.zeros_like(img, dtype='float')
    index = 0 if mode == 'v' else 1
    for prop in props:
        minIndex = np.min(np.transpose(prop['coords'])[index])
        maxIndex = np.max(np.transpose(prop['coords'])[index])
        for coord in prop['coords']:
            if coord[index] < prop['centroid'][index]:
                result[coord[0], coord[1]] = (coord[index] - minIndex) / \
                                             (prop['centroid'][index] - minIndex) - 1

            elif coord[index] > prop['centroid'][index]:
                result[coord[0], coord[1]] = (coord[index] - prop['centroid'][index]) / \
                                             (maxIndex - prop['centroid'][index])

    return result

def main(args):
    labels = sorted([i for i in os.listdir(args.path) if i.endswith('npy')])
    start = time.time()
    for i in tqdm(labels):
        img = np.load(Path(args.path) / i)[..., 0].squeeze().astype(np.uint32)
        props = regionprops(img)
        horizontalMap = mapGenarating(props, img, mode='h')
        verticalMap = mapGenarating(props, img, mode='v')
        if not Path(args.savePth).exists():
            os.makedirs(Path(args.savePth))

        # np.save(Path(args.savePth) / (i.split('.')[0] + '_horizontal.npy'), horizontalMap)
        # np.save(Path(args.savePth) / (i.split('.')[0] + '_vertical.npy'), verticalMap)
        plt.imshow(horizontalMap, cmap='jet')
        plt.show()
        plt.imshow(verticalMap, cmap='jet')
        plt.show()
        break
    print(f'Done! cost time: {time.time() - start}')

if __name__ == '__main__':
    args = parseArgs()
    main(args)

