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

def parseArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--rootPth', type=str, default=Path(__file__).parent.parent / 'data')
    parse.add_argument('--modelPth', type=str, default='../model/')

    return parse.parse_args()

def inference():
    pass

def main(args):
    pass

if __name__ == '__main__':
    args = parseArgs()
    inference()