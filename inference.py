# coding=utf-8
"""
@File   : inference.py
@Time   : 2020/01/10
@Author : Zengrui Zhao
"""
import argparse
import torch
from postProcess import proc
from metrics import *
from dataset import Data
from torch.utils.data import DataLoader
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import model

# device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
meanStd = ((0.80508233, 0.80461432, 0.8043749),(0.14636562, 0.1467832,  0.14712358))

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--rootPth', type=str, default=Path(__file__).parent.parent / 'data')
    parse.add_argument('--modelPth', type=str, default='../model/200114-154911/out_100.pth')

    return parse.parse_args()

def main(args):
    data = Data(root=Path(args.rootPth) / 'test',
                   mode='test',
                   isAugmentation=False)
    dataLoader = DataLoader(data)
    net = model().to(device)
    net.load_state_dict(torch.load(args.modelPth, map_location='cpu'))
    with torch.no_grad():
        for img, mask in dataLoader:
            img = img.to(device)
            mask = mask.squeeze()
            branchSeg, branchMSE = net(img)
            pred = torch.cat((branchSeg, branchMSE), dim=1).squeeze()
            output = proc(pred)
            plt.imshow(output, cmap='jet')
            plt.show()
            metricPQ, _ = get_fast_pq(mask, output)
            metricDice = get_dice_1(mask, output)
            print(f'Dice: {metricDice}, '
                  f'DQ: {metricPQ[0]}, '
                  f'SQ: {metricPQ[1]}, '
                  f'PQ: {metricPQ[2]}')

if __name__ == '__main__':
    args = parseArgs()
    main(args)