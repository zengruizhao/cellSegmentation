# coding=utf-8
"""
@File   : logger.py
@Time   : 2020/01/08
@Author : Zengrui Zhao
"""
import logging

def getLogger(logFile):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(logFile)
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

if __name__ == '__main__':
    logger = getLogger('./logs/log.log')