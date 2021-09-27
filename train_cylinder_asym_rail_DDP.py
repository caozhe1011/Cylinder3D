# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py


import os
import random
import time
import argparse
import sys
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.rail_metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import rail_data_builder_DDP, model_builder, loss_builder
from config.rail_config import load_config_data
from visualdl import LogWriter
import warnings
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

warnings.filterwarnings("ignore")


# seed = 310
# random.seed(seed)
# # 以下两句代码会牺牲一定的速度
# torch.backends.cudnn.deterministic = True  # GPU设置固定的随机数
# torch.backends.cudnn.benchmark = False  # torch自动查找快速方法，网络、输入输出不变的时候用
# np.random.seed(seed)
# torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)  # 设置GPU种子


def init_seeds(seed=1):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    # 以下两句代码会牺牲一定的速度
    torch.backends.cudnn.deterministic = True  # 设置卷积计算时的不变性
    torch.backends.cudnn.benchmark = False  # torch自动查找快速方法，网络、输入输出不变的时候用
    np.random.seed(seed)
    torch.manual_seed(seed)  # to seed the RNG for all devices (both CPU and CUDA)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)  # 多GPU时使用


def main(args):
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

    # todo:DDP初始化
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    # todo:DDP 设置随机种子
    rank = dist.get_rank()
    init_seeds(rank + 1)
    # todo:ddp
    my_model = model_builder.build(model_config)
    my_model.to(local_rank)
    # todo:DDP读取模型
    # DDP: Load模型要在构造DDP模型之前，并且两张卡都需要读取参数
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if os.path.exists(model_load_path):  # 读取check point
        # checkpoint = torch.load(model_load_path, map_location='cpu')
        checkpoint = torch.load(model_load_path, map_location=map_location)
        epoch = checkpoint['epoch']
        my_model.load_state_dict(checkpoint['model_state_dict'])
        best_val_miou = checkpoint['best_val_miou']
        print('load_checkpoint-->', model_save_path)
        print('epoch-{},best_val_miou-{}'.format(epoch, best_val_miou))
    else:
        epoch = 0
        best_val_miou = 0
        print('no_checkpoint')
        print('model_save_path-->', model_save_path)
    my_model = DDP(my_model, device_ids=[local_rank], output_device=local_rank)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    # 这里设置了ignore_label因此不会计算ignore_label的损失
    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True, num_class=num_class,
                                                   ignore_label=ignore_label)

    # 初始化Rail_sk类、初始化cylinder_dataset（附加数据增强） 并封装成dataloader
    train_dataset_loader, val_dataset_loaders = rail_data_builder_DDP.build(dataset_config,
                                                                            train_dataloader_config,
                                                                            val_dataloader_config,
                                                                            grid_size=grid_size)
    best_sum_miou = 0
    # training
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    with LogWriter(logdir="./log/train/new/base") as writer:
        while epoch < train_hypers['max_num_epochs']:
            # todo:DDP的dataloader
            # DDP：设置sampler的epoch，
            # DistributedSampler需要这个来指定shuffle方式，否则每一个epoch的shuffle结果一样
            train_dataset_loader.sampler.set_epoch(epoch)
            loss_list = []
            pbar = tqdm(total=len(train_dataset_loader), desc='rank' + str(dist.get_rank()))
            time.sleep(0.1)
            for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
                if global_iter % check_iter == 0 and epoch >= 1:
                    my_model.eval()  # 测试
                    val_loss_list = []
                    val_miou = []  # 记录3个测试集的平均iou（轨道+障碍物）
                    sum_miou = []  # 记录总体iou
                    with torch.no_grad():  # validation时两个进程都会
                        for i, val_dataset_loader in enumerate(val_dataset_loaders):
                            hist_list = []  # 混淆矩阵
                            pbar1 = tqdm(total=len(val_dataset_loader), desc='rank' + str(dist.get_rank()))
                            time.sleep(0.1)
                            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                                    val_dataset_loader):
                                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(local_rank) for i in
                                                  val_pt_fea]
                                val_grid_ten = [torch.from_numpy(i).to(local_rank) for i in val_grid]
                                val_label_tensor = val_vox_label.type(torch.LongTensor).to(local_rank)
                                # todo:为了防止dataset无法被batch size除尽的情况(最后一个mini batch不够batch size)
                                if len(val_pt_fea_ten) != val_batch_size:
                                    val_batch_size = len(val_pt_fea_ten)
                                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)  # 预测的体素label
                                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels), val_label_tensor,
                                                      ignore=ignore_label)
                                # loss += loss_func(predict_labels.detach(), val_label_tensor)
                                loss += loss_builder.geo_loss6(val_label_tensor, predict_labels, grid_size,
                                                               ignore_label, val_batch_size)
                                predict_labels = torch.argmax(predict_labels, dim=1)
                                predict_labels = predict_labels.cpu().detach().numpy()  # [1,480,360,32]
                                for count, i_val_grid in enumerate(val_grid):  # val_grid是每个点所在的体素index
                                    hist_list.append(fast_hist_crop(
                                        # 利用每个点的voxel_index,从预测得到的体素的label中取出每个点的label
                                        predict_labels[
                                            count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]],
                                        # 每个点的label，unique_label
                                        val_pt_labs[count], unique_label))
                                dist.all_reduce(loss)  # 将多卡的loss（sum运算）
                                val_loss_list.append(loss.detach().cpu().numpy() / dist.get_world_size())
                                pbar1.update(1)
                            pbar1.close()
                            # todo:DDP收集所需要的数据(两种方法将hist相加)
                            # 方法1 dist.all_reduce
                            hist = sum(hist_list)
                            hist = torch.tensor(hist, device=dist.get_rank())
                            dist.all_reduce(hist)  # 将多卡的hist（sum运算）
                            # print(dist.get_rank(),'----',hist)
                            iou = per_class_iu(hist.detach().cpu().numpy())

                            # 方法2 dist.all_gather
                            # dist.barrier()  # 等待多张卡eval完毕
                            # word_size = dist.get_world_size()
                            # outputs = torch.zeros_like(hist)
                            # output_tensors = [torch.zeros_like(hist) for _ in range(word_size)]  # 创建从多卡上收集到的tensor
                            # dist.all_gather(output_tensors, hist)  # 收集多卡中的hist_list
                            # for output in output_tensors:  # 每个进程都需要计算验证集的iou
                            #     outputs += output
                            # output_tensors = outputs.detach().cpu().numpy()
                            # iou = per_class_iu(output_tensors)  # 计算所有进程得到的混淆矩阵
                            del val_vox_label, val_grid, val_pt_fea, val_grid_ten, val_pt_labs
                            # 记录总体iou
                            sum_miou.append(np.nanmean(iou) * 100)
                            # 记录轨道iou
                            val_miou.append(np.nanmean((iou[1])) * 100)

                            if dist.get_rank() == 0:  # 只有显示需要在rank=0时
                                print('{}-Validation per class iou: '.format(val_folders[i]))
                                for class_name, class_iou in zip(unique_label_str, iou):
                                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                                print("{}- miou = {:.2f}".format(val_folders[i], np.nanmean(iou) * 100))
                                print("{}- loss = {:.2f}".format(val_folders[i], np.nanmean(val_loss_list)))

                    my_model.train()
                    # save model if performance is improved
                    val_miou = np.asarray(val_miou)
                    val_miou = np.nanmean(val_miou)
                    sum_miou = np.asarray(sum_miou)
                    sum_miou = np.nanmean(sum_miou)

                    # writer.add_scalar(tag="val_miou", step=global_iter, value=float(val_miou))
                    if best_sum_miou < sum_miou:
                        best_sum_miou = sum_miou
                        if dist.get_rank() == 0:
                            state = {'epoch': epoch,
                                     'model_state_dict': my_model.module.state_dict(),
                                     'best_val_miou': best_sum_miou}
                            # model_save_path = model_save_path.split('.pt')[0] + '_best_miou.pt'
                            torch.save(state, model_save_path.split('.pt')[0] + '_best_miou.pt')
                            print("save--model--best_miou-->"+model_save_path)
                    if dist.get_rank() == 0:
                        print('Current val miou is %.3f while the best val miou is %.3f' % (sum_miou, best_sum_miou))

                    if best_val_miou < val_miou:
                        best_val_miou = val_miou
                        # todo:DDP保存模型
                        # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
                        #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
                        # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
                        if dist.get_rank() == 0:
                            state = {'epoch': epoch,
                                     'model_state_dict': my_model.module.state_dict(),
                                     'best_val_miou': best_val_miou}
                            torch.save(state, model_save_path)
                            print("save--model-->"+model_save_path)
                    if dist.get_rank() == 0:
                        print('Current rail miou is %.3f while the best rail miou is %.3f' %
                              (val_miou, best_val_miou))
                        print('Current val loss is %.3f' %
                              (np.mean(val_loss_list)))
                # [相对坐标，xy的极坐标、高度，xy笛卡尔坐标,反射率],[N,9]
                train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(local_rank) for i in train_pt_fea]
                # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
                train_vox_ten = [torch.from_numpy(i).to(local_rank) for i in train_grid]  # [N,3]每个点在体素中的位置[N,3]
                point_label_tensor = train_vox_label.type(torch.LongTensor).to(local_rank)

                # forward + backward + optimize
                # todo:为了防止dataset无法被batch size除尽的情况(最后一个mini batch不够batch size)
                if len(train_pt_fea_ten) != train_batch_size:
                    train_batch_size = len(train_pt_fea_ten)
                outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)  # [B=1,3,480,360,32] 每一个体素所代表的类别
                loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=ignore_label)
                # loss += loss_func(outputs, point_label_tensor)
                # TODO:新加的loss
                loss += loss_builder.geo_loss6(point_label_tensor, outputs, grid_size, ignore_label, train_batch_size)
                # writer.add_scalar(tag="loss", step=global_iter, value=loss)
                loss.backward()
                optimizer.step()

                loss_list.append(loss.item())  # 记录多次迭代的loss

                if global_iter % check_iter == 0:  # 1000次迭代后输出平均loss
                    if len(loss_list) > 0:
                        print('epoch %d iter %5d, loss: %.3f, rank:%d\n' %
                              (epoch, i_iter, np.mean(loss_list), dist.get_rank()))
                    else:
                        print('loss error')

                optimizer.zero_grad()
                pbar.update(1)
                global_iter += 1

            pbar.close()
            epoch += 1


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/rail.yaml')
    # todo:DDP会自动分发local rank
    parser.add_argument("--local_rank", default=-1, type=int)
    args = parser.parse_args()

    print(' '.join(sys.argv))  # argv：用户输入的参数
    print(args)
    main(args)
