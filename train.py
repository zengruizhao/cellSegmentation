# coding=utf-8
"""
@File   : train.py
@Time   : 2020/01/08
@Author : Zengrui Zhao
"""
import torch
import torch.nn as nn
import numpy as np
import argparse
from pathlib import Path
from logger import getLogger
import time
import os.path as osp
import os
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from dataset import Data, getGradient
from torch.utils.data import DataLoader
from ranger import Ranger
from model import model
import sys
import segmentation_models_pytorch.utils.losses as smploss

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=100)
    parse.add_argument('--batchsizeTrain', type=int, default=2)
    parse.add_argument('--batchsizeTest', type=int, default=16)
    parse.add_argument('--rootPth', type=str, default=Path(__file__).parent.parent / 'data')
    parse.add_argument('--logPth', type=str, default='../log')
    parse.add_argument('--numWorkers', type=int, default=14)
    parse.add_argument('--evalFrequency', type=int, default=1)
    parse.add_argument('--saveFrequency', type=int, default=1)
    parse.add_argument('--msgFrequency', type=int, default=5)
    parse.add_argument('--tensorboardPth', type=str, default='../tensorboard')
    parse.add_argument('--modelPth', type=str, default='../model')

    return parse.parse_args()

def main(args, logger):
    writter = SummaryWriter(logdir=args.subTensorboardPth)
    trainSet = Data(root=Path(args.rootPth) / 'train',
                    mode='train',
                    isAugmentation=True,
                    cropSize=(512, 512))
    trainLoader = DataLoader(trainSet,
                             batch_size=args.batchsizeTrain,
                             shuffle=True,
                             pin_memory=False,
                             drop_last=False,
                             num_workers=args.numWorkers)
    testSet = Data(root=Path(args.rootPth) / 'test',
                    mode='test',
                    isAugmentation=False)
    testLoader = DataLoader(testSet,
                             batch_size=args.batchsizeTest,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False,
                             num_workers=args.numWorkers)
    net = model().to(device)
    criterionMSE = nn.MSELoss().to(device)
    criterionDice = smploss.DiceLoss(eps=sys.float_info.min).to(device)
    criterionCE = nn.CrossEntropyLoss().to(device)
    optimizer = Ranger(net.parameters(), lr=.01)
    runningLoss = []
    iter = 0
    for epoch in range(args.epoch):
        for img, mask, horizontalVertical in trainLoader:
            iter += 1
            img, mask, horizontalVertical = img.to(device), mask.to(device), horizontalVertical.to(device)
            optimizer.zero_grad()
            [branchSeg, branchMSE] = net(img)
            predictionGradient = getGradient(branchMSE)
            gtGradient = getGradient(horizontalVertical)
            #
            # gtGradient1 = gtGradient.cpu().detach().numpy()[0, ...]
            # horizontalVertical1 = horizontalVertical.cpu().detach().numpy()[0, ...]
            # fig, axes = plt.subplots(2, 2)
            # axes[0, 0].imshow(gtGradient1[1, ...], cmap='jet')
            # axes[0, 1].imshow(horizontalVertical1[1, ...], cmap='jet')
            # axes[1, 0].imshow(gtGradient1[0, ...], cmap='jet')
            # axes[1, 1].imshow(horizontalVertical1[0, ...], cmap='jet')
            # axes[0, 0].set_title('Gradient', fontsize=20)
            # axes[0, 0].set_ylabel('Horizontal', fontsize=20)
            # axes[0, 1].set_title('Raw', fontsize=20)
            # axes[1, 0].set_ylabel('Vertical', fontsize=20)
            # plt.show()
            loss = criterionMSE(branchMSE, horizontalVertical) + \
                   criterionMSE(predictionGradient, gtGradient) + \
                   criterionDice(branchSeg[:, 1, ...], mask) + \
                   criterionCE(branchSeg, mask.long())
            loss.backward()
            optimizer.step()
            runningLoss.append(loss.item())
            if iter % args.msgFrequency == 0:
                logger.info(f'epoch:{epoch}/{args.epoch}, '
                            f'loss:{np.mean(runningLoss)}')

                writter.add_scalar('Loss', np.mean(runningLoss))
                runningLoss = []

            if epoch % args.evalFrequency == 0:
                pass


if __name__ == '__main__':
    args = parseArgs()
    uniqueName = time.strftime('%y%m%d-%H%M%S')
    args.subModelPth = osp.join(args.modelPth, uniqueName)
    args.subTensorboardPth = osp.join(args.tensorboardPth, uniqueName)
    for subDir in [args.logPth,
                   args.subModelPth,
                   args.subTensorboardPth]:
        if not osp.exists(subDir):
            os.makedirs(subDir)

    logFile = osp.join(args.logPth, uniqueName + '.log')
    logger = getLogger(logFile)
    for k, v in args.__dict__.items():
        logger.info(k)
        logger.info(v)

    main(args, logger)