# -*- coding:utf-8 -*-
# author: Xinge
# @file: data_builder.py 

import torch
from dataloader.dataset_rail import get_model_class, collate_fn_BEV, collate_fn_BEV_test
from dataloader.pc_dataset import get_pc_model_class
import torch.utils.data


def build(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[480, 360, 32],
          return_zyx=False):
    train_data_path = train_dataloader_config["data_path"]
    train_imageset = train_dataloader_config["imageset"]
    val_data_path = val_dataloader_config["data_path"]
    val_imageset = val_dataloader_config["imageset"]
    train_ref = train_dataloader_config["return_ref"]
    val_ref = val_dataloader_config["return_ref"]

    label_mapping = dataset_config["label_mapping"]
    val_folders = dataset_config["val_folders"]  # [a3,a6,a7]

    rail = get_pc_model_class(dataset_config['pc_dataset_type'])  # 初始化Rail_sk类

    train_pt_dataset = rail(train_data_path, imageset=train_imageset,
                            return_ref=train_ref, label_mapping=label_mapping, nusc=None)
    val_pt_datasets = []
    for i in val_folders:
        val_pt_datasets.append(rail(val_data_path, imageset=val_imageset, return_ref=val_ref,
                                    label_mapping=label_mapping, val_folder=i, nusc=None))
    # val_pt_dataset = rail(data_path, imageset=val_imageset,
    #                       return_ref=val_ref, label_mapping=label_mapping, val_folder=i, nusc=None)

    train_dataset = get_model_class(dataset_config['dataset_type'])(  # 初始化cylinder_dataset
        train_pt_dataset,
        grid_size=grid_size,
        # flip_aug=True,
        flip_aug=False,  # 随机取一个轴的坐标为相反数，根据一个轴进行镜像变化（因为轨道不是360度所以不能用）
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        ignore_label=dataset_config["ignore_label"],
        rotate_aug=True,  # 随机旋转
        scale_aug=True,  # 随机缩放
        transform_aug=True  # 随机增加高斯噪声
    )
    val_datasets = []
    for val_pt_dataset in val_pt_datasets:
        val_datasets.append(get_model_class(dataset_config['dataset_type'])(
            val_pt_dataset,
            grid_size=grid_size,
            fixed_volume_space=dataset_config['fixed_volume_space'],
            max_volume_space=dataset_config['max_volume_space'],
            min_volume_space=dataset_config['min_volume_space'],
            ignore_label=dataset_config["ignore_label"],
            return_test=return_zyx
        ))
    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=train_dataloader_config["batch_size"],
                                                       collate_fn=collate_fn_BEV,
                                                       shuffle=train_dataloader_config["shuffle"],
                                                       num_workers=train_dataloader_config["num_workers"])
    val_dataset_loaders = []
    if return_zyx is True:  # test时需要返回原先的xyz
        for val_dataset in val_datasets:
            val_dataset_loaders.append(torch.utils.data.DataLoader(dataset=val_dataset,
                                                                   batch_size=val_dataloader_config["batch_size"],
                                                                   collate_fn=collate_fn_BEV_test,
                                                                   shuffle=val_dataloader_config["shuffle"],
                                                                   num_workers=val_dataloader_config["num_workers"]))
    else:
        for val_dataset in val_datasets:
            val_dataset_loaders.append(torch.utils.data.DataLoader(dataset=val_dataset,
                                                                   batch_size=val_dataloader_config["batch_size"],
                                                                   collate_fn=collate_fn_BEV,
                                                                   shuffle=val_dataloader_config["shuffle"],
                                                                   num_workers=val_dataloader_config["num_workers"]))

    return train_dataset_loader, val_dataset_loaders
