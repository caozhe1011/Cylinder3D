# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py

import random
import torch
import numpy as np
import os
import time
import argparse
import sys
import torch.optim as optim
from tqdm import tqdm

from utils.rail_metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import rail_data_builder, model_builder, loss_builder
from config.rail_config import load_config_data

from utils.load_save_util import load_checkpoint
from visualdl import LogWriter

import warnings

warnings.filterwarnings("ignore")

# myseed = 42069  # set a random seed for reproducibility
# random.seed(myseed)
# torch.backends.cudnn.deterministic = True  # GPU设置固定的随机数
# torch.backends.cudnn.benchmark = False  # torch自动查找快速方法，网络、输入输出不变的时候用
# np.random.seed(myseed)
# torch.manual_seed(myseed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(myseed)  # 多GPU时使用

seed = 42069
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)  # 读取配置文件

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']

    model_load_path = train_hypers['model_load_path']
    model_save_path = train_hypers['model_save_path']

    val_folders = val_dataloader_config['val_folders']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]  # 找出unique_label所对应的class

    my_model = model_builder.build(model_config)


    if os.path.exists(model_load_path):  # 读取check point
        # my_model = load_checkpoint(model_load_path, my_model)
        checkpoint = torch.load(model_load_path)
        epoch = checkpoint['epoch']
        my_model.load_state_dict(checkpoint['model_state_dict'])
        print('load_checkpoint-->', model_save_path)
    else:
        epoch = 0
        print('no_checkpoint')
        print('model_save_path-->', model_save_path)
    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    # 这里设置了ignore_label因此不会计算ignore_label的损失
    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True, num_class=num_class,
                                                   ignore_label=ignore_label)

    # 初始化Rail_sk类、初始化cylinder_dataset（附加数据增强） 并封装成dataloader
    train_dataset_loader, val_dataset_loaders = rail_data_builder.build(dataset_config,
                                                                        train_dataloader_config,
                                                                        val_dataloader_config,
                                                                        grid_size=grid_size)

    # training

    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    with LogWriter(logdir="./log/train/block") as writer:
        while epoch < train_hypers['max_num_epochs']:
            loss_list = []
            pbar = tqdm(total=len(train_dataset_loader))
            time.sleep(0.1)
            for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
                if global_iter % check_iter == 0 and epoch >= 1:
                    my_model.eval()  # 测试
                    hist_list = []
                    val_loss_list = []
                    val_miou = []  # 记录3个测试集的平均iou
                    with torch.no_grad():
                        for i, val_dataset_loader in enumerate(val_dataset_loaders):
                            pbar1 = tqdm(total=len(val_dataset_loader))
                            time.sleep(0.1)
                            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                                    val_dataset_loader):
                                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i
                                                  in
                                                  val_pt_fea]
                                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
                                if len(val_pt_fea_ten) != val_batch_size:  # todo:为了防止dataset无法被batch size除尽的情况
                                    val_batch_size = len(val_pt_fea_ten)
                                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)  # 预测的体素label
                                # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels), val_label_tensor,
                                                      ignore=ignore_label)
                                loss += loss_func(predict_labels.detach(), val_label_tensor)
                                # loss += loss_builder.geo_loss(val_label_tensor, predict_labels, grid_size, ignore_label, val_batch_size)
                                predict_labels = torch.argmax(predict_labels, dim=1)
                                predict_labels = predict_labels.cpu().detach().numpy()  # [1,480,360,32]
                                for count, i_val_grid in enumerate(val_grid):  # val_grid是每个点所在的体素index
                                    hist_list.append(fast_hist_crop(
                                        # 利用每个点的voxel_index,从预测得到的体素的label中取出每个点的label
                                        predict_labels[
                                            count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                                        # 每个点的label，unique_label
                                        val_pt_labs[count], unique_label))
                                val_loss_list.append(loss.detach().cpu().numpy())
                                pbar1.update(1)
                            pbar1.close()
                            iou = per_class_iu(sum(hist_list))
                            print('{}-Validation per class iou: '.format(val_folders[i]))
                            for class_name, class_iou in zip(unique_label_str, iou):
                                print('%s : %.2f%%' % (class_name, class_iou * 100))
                            print("{}- miou = {}".format(val_folders[i], np.nanmean(iou) * 100))
                            val_miou.append(np.nanmean(iou) * 100)
                            del val_vox_label, val_grid, val_pt_fea, val_grid_ten
                    my_model.train()
                    # save model if performance is improved
                    val_miou = np.asarray(val_miou)
                    val_miou = np.nanmean(val_miou)
                    writer.add_scalar(tag="val_miou", step=global_iter, value=float(val_miou))
                    if best_val_miou < val_miou:
                        best_val_miou = val_miou
                        state = {'epoch': epoch,
                                 'model_state_dict': my_model.state_dict()}
                        torch.save(state, model_save_path)
                        print("save--model")
                    print('Current val miou is %.3f while the best val miou is %.3f' %
                          (val_miou, best_val_miou))
                    print('Current val loss is %.3f' %
                          (np.mean(val_loss_list)))
                # [相对坐标，xy的极坐标、高度，xy笛卡尔坐标,反射率],[N,9]
                train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                    train_pt_fea]
                # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
                train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]  # [N,3]每个点在体素中的位置[N,3]
                point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)

                # forward + backward + optimize
                if len(train_pt_fea_ten) != train_batch_size:  # todo:为了防止dataset无法被batch size除尽的情况
                    train_batch_size = len(train_pt_fea_ten)
                outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)  # [B=1,3,480,360,32] 每一个体素所代表的类别
                # loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor,
                #                       ignore=ignore_label) + loss_func(outputs, point_label_tensor)
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=ignore_label)
                loss += loss_func(outputs, point_label_tensor)
                # TODO:新加的loss
                # loss += loss_builder.geo_loss(point_label_tensor, outputs, grid_size, ignore_label, train_batch_size)
                # if torch.isnan(loss):
                #     print(global_iter)
                writer.add_scalar(tag="loss", step=global_iter, value=loss)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.item())  # 记录多次迭代的loss
                optimizer.zero_grad()
                pbar.update(1)
                global_iter += 1
                if global_iter % check_iter == 0:
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f\n' %
                              (epoch, i_iter, np.mean(loss_list)))
                    else:
                        print('loss error')
            pbar.close()
            epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/rail.yaml')
    args = parser.parse_args()

    print(' '.join(sys.argv))  # argv：用户输入的参数
    print(args)
    main(args)
