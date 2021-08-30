# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py 

import numpy as np


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    bin_count = np.bincount(n * label[k].astype(int) + pred[k], minlength=n ** 2)  # n用来调整行数，一行代表一个类别的所有预测情况
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))  # 混淆矩阵计算iou


def fast_hist_crop(output, target, unique_label):
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)  # +2是因为是从0开始并且加上ignore的一类
    hist = hist[unique_label + 1, :]  # 把ignore部分去掉（0标签的去掉）
    hist = hist[:, unique_label + 1]
    return hist
