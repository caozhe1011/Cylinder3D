import os
import time
import argparse
import sys
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
from tqdm import tqdm

from utils.rail_metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import rail_data_builder, model_builder, loss_builder
from config.rail_config import load_config_data
import warnings
from visual_test import visual_3d


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path

    configs = load_config_data(config_path)  # 读取配置文件

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']

    model_config = configs['model_params']

    grid_size = model_config['output_shape']

    model_load_path = args.model_load_path

    val_folders = val_dataloader_config['val_folders']
    save_root_path = args.save_root_path
    # val_folders = ['test']

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]  # 找出unique_label所对应的class

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):  # 读取check point
        checkpoint = torch.load(model_load_path, map_location='cpu')
        my_model.load_state_dict(checkpoint['model_state_dict'])
        print('load_checkpoint')

    my_model.to(pytorch_device)

    # 初始化Rail_sk类、初始化cylinder_dataset（附加数据增强） 并封装成dataloader
    _, val_dataset_loaders = rail_data_builder.build(dataset_config,
                                                     train_dataloader_config,
                                                     val_dataloader_config,
                                                     grid_size=grid_size, return_zyx=True)

    # evaluate
    my_model.eval()  # 测试
    hist_list = []
    with torch.no_grad():
        for i, val_dataset_loader in enumerate(val_dataset_loaders):
            if val_folders[i] != 'x':
                pbar1 = tqdm(total=len(val_dataset_loader))
                time.sleep(0.1)
                if not os.path.exists(os.path.join(save_root_path, val_folders[i])):
                    os.makedirs(os.path.join(save_root_path, val_folders[i]))
                for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, zyx) in enumerate(
                        val_dataset_loader):
                    save_path = os.path.join(save_root_path, val_folders[i], str(i_iter_val))
                    if i_iter_val < 0:
                        pbar1.update(1)
                        continue
                    val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                      val_pt_fea]
                    val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]

                    predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                    predict_labels = torch.argmax(predict_labels, dim=1)
                    predict_labels = predict_labels.cpu().detach().numpy()

                    for count, i_val_grid in enumerate(val_grid):
                        pre_point_label = predict_labels[
                            count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]
                        hist_list.append(fast_hist_crop(
                            # 利用每个点的voxel_index,从预测得到的体素的label中取出每个点的label
                            pre_point_label,
                            # label，unique_label
                            val_pt_labs[count], unique_label))
                        one_iou = per_class_iu(fast_hist_crop(pre_point_label, val_pt_labs[count], unique_label))
                        print(one_iou)
                        zyx_one = zyx[count]
                        zyx_one[:, [0, 2]] = zyx_one[:, [2, 0]]
                        xyz = zyx_one
                        evaluation = np.concatenate((xyz, pre_point_label[:, np.newaxis]), axis=1)
                        np.save(save_path, evaluation)
                    # batch_size 只能是1，在可视化时
                    '''
                    zyx = zyx[0]
                    zyx[:, [0, 2]] = zyx[:, [2, 0]]
                    xyz = zyx
                    evaluation = np.concatenate((xyz, pre_point_label[:, np.newaxis]), axis=1)
                    point_label = val_pt_labs[0]
                    original = np.concatenate((xyz, point_label), axis=1)
                    np.save('./original', original)
                    np.save('./eval.npy', evaluation)
                    visual_3d(visual_flag="rail", ori_path='./original.npy', eval_path='./eval.npy')
                    np.save(save_path, evaluation)
                    '''
                    pbar1.update(1)
                pbar1.close()
                iou = per_class_iu(sum(hist_list))
                print('{}-Validation per class iou: '.format(val_folders[i]))
                for class_name, class_iou in zip(unique_label_str, iou):
                    print('%s : %.2f%%' % (class_name, class_iou * 100))
                del val_vox_label, val_grid, val_pt_fea, val_grid_ten


if __name__ == '__main__':
    # Training settings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--config_path', default='config/rail.yaml')  # 读取体素分割大小、label、batchsize、验证集文件夹
    parser.add_argument('--model_load_path', default='./model_save_dir/block_geo6_sparse.pt')
    parser.add_argument('--save_root_path', default='./test')
    args = parser.parse_args()

    print(' '.join(sys.argv))  # argv：用户输入的参数
    print(args)
    main(args)
