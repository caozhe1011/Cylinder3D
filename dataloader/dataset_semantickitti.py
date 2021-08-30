# -*- coding:utf-8 -*-
# author: Xinge

"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data
import pickle

REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


@register_dataset
class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3]):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any(): print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # process labels
        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])  # 求出弧度
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)  # 拼接上Z坐标


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


@register_dataset
class cylinder_dataset(data.Dataset):
    """
    数据增强：
    rotate_aug 随机旋转[-45,45]
    flip_aug 随机改变一列为相反数
    scale_aug 随机缩放±0.05
    transform_aug trans_std=[0.1, 0.1, 0.1]  随机增加高斯噪声
    """

    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]  # 调用Rail_sk的getitem函数
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data  # xyz，label，反射率
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4  # 将角度从度转换为弧度deg2rad，随机旋转[-45,45]
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.asarray([[c, s], [-s, c]])  # 旋转矩阵
            xyz[:, :2] = np.dot(xyz[:, :2], j)  # todo：是不是应该左乘旋转矩阵，只对xy坐标进行旋转

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)  # 随机改变一列为相反数
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)  # 随机缩放±0.05
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        if self.transform:  # trans_std=[0.1, 0.1, 0.1]  随机增加高斯噪声
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        # convert coordinate into polar coordinates 转极坐标
        xyz_pol = cart2polar(xyz)  # 对xy进行极坐标转换，返回(rho, phi, Z坐标)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)  # 求最大值半径rho
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)  # 求最小值半径
        max_bound = np.max(xyz_pol[:, 1:], axis=0)  # 求最大角度和z
        min_bound = np.min(xyz_pol[:, 1:], axis=0)  # 求最小角度和z
        max_bound = np.concatenate(([max_bound_r], max_bound))  # 最大值半径rho,最大角度,z
        min_bound = np.concatenate(([min_bound_r], min_bound))
        # max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4]
        if self.fixed_volume_space:  # todo：将bound人为设定,用于排除离群点
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size  # grid_size=[480, 360, 32][半径、角度、z]
        intervals = crop_range / (cur_grid_size - 1)  # 计算间隔[半径、角度、z]

        if (intervals == 0).any():
            print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(
            np.int)  # 算出每个点在哪一个体素中[N,3]    clip将不再范围内的点划入(min_bound, max_bound)范围

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1  # dim_array=[-1,1,1,1]
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(
            dim_array)  # [3, 480, 360, 32]表示每一个体素的起始位置，（体素index * 间隔 + 起始距离）
        voxel_position = polar2cat(voxel_position)  # 从极坐标转换为笛卡尔坐标返回每一个体素的起始位置

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label  # [480, 360, 32]
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)  # 记录体素index和labels对应的关系[grid_ind, labels],[N,4]
        # 根据xyz三条件排序，z的优先级最高，y其次，x最后
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)  # 计算体素的label
        data_tuple = (voxel_position, processed_label)  # [480,360,32] processed_label返回每个体素中最多的label，来表示体素的label

        # center data on each voxel for PTnet
        # todo: 算出voxel中心，为什么+0.5？因为之前grid_ind向下取整
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers  # 相对于中心的坐标
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)  # [相对坐标，xy的极坐标、高度，xy笛卡尔坐标] [N,8]

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)  # 拼接上反射率 [N,9]

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            # (voxel_position, processed_label)+=(grid_ind, labels, return_fea)5个tuple
            # [3,480,360,32]\[480,360,32]\[N,3]\[N,1]\[N,9]
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


@register_dataset
class polar_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 45) - np.pi / 8
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        return data_tuple


# @nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)  # jit加速
# processed_label[480, 360, 32]的0阵  ,  sorted_label_voxel_pair[N,4] 每个点在体素的index和label
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)  # 统计label数量
    counter[sorted_label_voxel_pair[0, 3]] = 1  # counter[第一个点的label] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]  # 第一个点在体素中的index
    '''
     因为输入的是排序过后的label，因此可以直接从头到尾判断index是否相同，统计每个体素内的label有多少 ,这里能够减少一个for循环，加快了速度
    '''
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]  # 第i点在体素中的index
        if not np.all(np.equal(cur_ind, cur_sear_ind)):  # 判断两点的体素index是否相同，若不同
            # processed_label[体素index]=counter中最大值的索引
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)  # 重新置零
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1  # 两点的体素index若相同，counter[第i个点的label] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label  # [480,360,32] 返回每个体素中最多的label，来表示体素的label


def collate_fn_BEV(data):  # 因为设置的batch size=1 所以不用根据每一个batch size进行处理
    # data:(voxel_position, processed_label,grid_ind, labels, return_fea)5个tuple
    # [3,480,360,32]\[480,360,32]\[N,3]\[N,1]\[N,9]
    # return_fea:[相对坐标，xy的极坐标、高度，xy笛卡尔坐标，反射率] [N,9]
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    point_fea = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, point_fea


def collate_fn_BEV_test(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    point_fea = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, point_fea, index
