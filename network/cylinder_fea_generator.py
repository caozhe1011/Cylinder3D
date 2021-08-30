# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class cylinder_fea(nn.Module):

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):
        super(cylinder_fea, self).__init__()

        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        self.max_pt = max_pt_per_encode
        self.fea_compre = fea_compre
        self.grid_size = grid_size
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)
        self.pool_dim = out_pt_fea_dim

        # point feature compression
        if self.fea_compre is not None:
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())
            self.pt_fea_dim = self.fea_compre
        else:
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind):
        # pt_fea [相对坐标，xy的极坐标、高度，xy笛卡尔坐标,反射率]，[N,9]的一个list
        # xy_ind [N,3]每个点在体素中的位置，[N,3]的一个list
        cur_dev = pt_fea[0].get_device()  # 查询device编号

        # concate everything
        cat_pt_ind = []
        for i_batch in range(len(xy_ind)):
            # 最后一维度左填充1列的i_batch常数(为了之后稀疏卷积的构造coor，添加上了batch的编号)
            cat_pt_ind.append(F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))  # [N,4]
        # 将多个batch的数据拼接
        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)
        pt_num = cat_pt_ind.shape[0]

        # shuffle the data
        shuffled_ind = torch.randperm(pt_num, device=cur_dev)  # 打乱得到[0,pt_num)的tensor放入device中
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # unique xy grid index,
        # N`原始始输入点的个数，N降采样后个数
        # unq(求出圆柱形体素内有点存在的体素块idx)  [N,4]
        # unq_cnt(对应体素块内的点的个数)[N]
        # unq_inv[N`]，
        # unq:不重复的元素(升序排列)  unq_inv:不重复元素在原先tensor中的index即cat_pt_ind = uni[unq_inv]  unq_cnt:不重复元素数量
        unq, unq_inv, unq_cnt = torch.unique(cat_pt_ind, return_inverse=True, return_counts=True, dim=0)
        unq = unq.type(torch.int64)

        # process feature，升维点特征
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)
        pooled_data = torch_scatter.scatter_max(processed_cat_pt_fea, unq_inv, dim=0)[0]  # 求出每一个体素内的点特征的最大值(降采样)

        if self.fea_compre:  # 将点特征压缩到fea_compre维度 16
            processed_pooled_data = self.fea_compression(pooled_data)
        else:
            processed_pooled_data = pooled_data

        return unq, processed_pooled_data  # unq(求出圆柱形体素内有点存在的体素块idx)，processed_pooled_data(体素降采样后的点特征)
