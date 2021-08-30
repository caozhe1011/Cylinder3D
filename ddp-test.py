#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/6/22 22:00
# @Author  : Young
# @File    : ddp-test.py
import os
import time

os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
import torch
import torch.distributed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

torch.distributed.init_process_group(backend="nccl")

batch_size = 4
data_size = 8

local_rank = torch.distributed.get_rank()
print(local_rank)
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)


class RandomDataset(Dataset):
    def __init__(self, length, local_rank):
        self.len = length
        self.data = torch.stack(
            [torch.ones(1), torch.ones(1) * 2, torch.ones(1) * 3, torch.ones(1) * 4, torch.ones(1) * 5,
             torch.ones(1) * 6, torch.ones(1) * 7, torch.ones(1) * 8]).to('cuda')
        self.local_rank = local_rank

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


dataset = RandomDataset(data_size, local_rank)
sampler = DistributedSampler(dataset)
# 测试普通dataloader和DistributedSampler的区别
# rand_loader =DataLoader(dataset=dataset,batch_size=batch_size,sampler=None,shuffle=True)
rand_loader = DataLoader(dataset=dataset, batch_size=batch_size, sampler=sampler)
epoch = 0
while epoch < 2:
    print('-------------{}-start---------------'.format(epoch))
    sampler.set_epoch(epoch)  # 若不在每次epoch前设置随机种子，那么每一个epoch的data都是一样的
    for data in rand_loader:
        print(data)
        # output_tensors = [data.clone() for _ in range(torch.distributed.get_world_size())]
        # torch.distributed.all_gather(output_tensors,data)
        # print('gather',output_tensors)
    print('-------------{}-end---------------'.format(epoch))
    epoch += 1