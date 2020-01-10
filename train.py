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
import torch.nn.functional as F
import torch.nn as nn
# from inference import inference
from postProcess import proc
from metrics import *

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--epoch', type=int, default=300)
    parse.add_argument('--batchsizeTrain', type=int, default=2)
    parse.add_argument('--batchsizeTest', type=int, default=2)
    parse.add_argument('--rootPth', type=str, default=Path(__file__).parent.parent / 'data')
    parse.add_argument('--logPth', type=str, default='../log')
    parse.add_argument('--numWorkers', type=int, default=14)
    parse.add_argument('--evalFrequency', type=int, default=50)
    parse.add_argument('--saveFrequency', type=int, default=1)
    parse.add_argument('--msgFrequency', type=int, default=5)
    parse.add_argument('--tensorboardPth', type=str, default='../tensorboard')
    parse.add_argument('--modelPth', type=str, default='../model')

    return parse.parse_args()

def eval(net, dataloader, logger):
    dq, sq, pq, aji, dice2 = [], [], [], [], []
    with torch.no_grad():
        for img, mask in dataloader:
            img = img.to(device)
            branchSeg, branchMSE = net(img)
            pred = torch.cat((branchSeg, branchMSE), dim=1)
            for i in range(pred.shape[0]):
                output = proc(pred[i, ...])
                # plt.imshow(output, cmap='jet')
                # plt.show()
                metricPQ, _ = get_fast_pq(mask[i, ...], output)
                metricAJI = get_fast_aji_plus(mask[i, ...], output)
                metricDice2 = get_fast_dice_2(mask[i, ...], output)
                dq.append(metricPQ[0])
                sq.append(metricPQ[1])
                pq.append(metricPQ[-1])
                aji.append(metricAJI)
                dice2.append(metricDice2)

    logger.info(f'dq: {np.mean(dq)}, '
                f'sq: {np.mean(sq)}, '
                f'pq: {np.mean(pq)}, '
                f'AJI: {np.mean(aji)}, '
                f'Dice2: {np.mean(dice2)}')

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
    # net = nn.DataParallel(net)
    criterionMSE = nn.MSELoss().to(device)
    criterionDice = smploss.DiceLoss(eps=1e-7).to(device)
    # criterionCE = nn.CrossEntropyLoss().to(device)
    criterionCE = nn.BCELoss().to(device)
    optimizer = Ranger(net.parameters(), lr=1.e-1)
    runningLoss, MSEloss, CEloss, Diceloss = [], [], [], []
    iter = 0
    for epoch in range(args.epoch):
        if epoch != 0 and epoch % args.evalFrequency == 0:
            logger.info(f'===============Eval after epoch {epoch}...===================')
            eval(net, testLoader, logger)

        for img, mask, horizontalVertical in trainLoader:
            iter += 1
            img, mask, horizontalVertical = img.to(device), mask.to(device), horizontalVertical.to(device)
            optimizer.zero_grad()
            [branchSeg, branchMSE] = net(img)
            predictionGradient = getGradient(branchMSE)
            gtGradient = getGradient(horizontalVertical)

            loss1 = criterionMSE(branchMSE, horizontalVertical) + \
                   2. * criterionMSE(predictionGradient, gtGradient)
            loss2 = criterionCE(branchSeg, mask)
            loss3 = criterionDice(branchSeg, mask)
            loss = loss1 + loss2 + loss3

            loss.backward()
            optimizer.step()
            MSEloss.append(loss1.item())
            CEloss.append(loss3.item())
            Diceloss.append(loss2.item())
            runningLoss.append(loss.item())
            if iter % args.msgFrequency == 0:
                logger.info(f'epoch:{epoch}/{args.epoch}, '
                            f'loss:{np.mean(runningLoss):.4f}, '
                            f'MSEloss:{np.mean(MSEloss):.4f}, '
                            f'Diceloss:{np.mean(Diceloss):.4f}, '
                            f'CEloss:{np.mean(CEloss):.4f}')

                # writter.add_scalar('Loss', np.mean(runningLoss))
                runningLoss, MSEloss, CEloss, Diceloss = [], [], [], []

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