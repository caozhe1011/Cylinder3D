'''
最终读取模型，并实时显示
'''
import os
import time
import argparse
import sys
import numpy as np
import torch
from dataloader.pc_dataset import get_SemKITTI_label_name
from builder import rail_data_builder, model_builder
from config.rail_config import load_config_data
import warnings
import open3d as o3d
# import win32gui
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui, QtWidgets
from ui.driver_ui import Ui_MainWindow
from multiprocessing import Queue, Process
import multiprocessing as mp


def pc_colors(arr: np.ndarray):
    arr = arr.astype(np.int)
    color_list = np.asarray([
        [127, 127, 127],
        [0, 255, 0],
        # [0, 0, 255],
        [127, 127, 127],
        [238, 48, 167],
        [127, 127, 127],
    ])
    colors = []
    for i in arr:
        colors.append(color_list[i])

    return np.asarray(colors) / 255


def load_data(pcl: o3d.geometry.PointCloud, data_root, data_path):
    data_path = os.path.join(data_root, data_path)
    data = np.load(data_path)
    pcl.points = o3d.utility.Vector3dVector(data[:, 0:3])
    pcl.colors = o3d.utility.Vector3dVector(pc_colors(data[:, -1]))
    return pcl, data


def create_visualizer(name, left=0, top=0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(name, left=left, width=700, height=900, top=top)
    pcl = o3d.geometry.PointCloud()
    block_pcl = o3d.geometry.PointCloud()
    line_set = o3d.geometry.LineSet()
    vis.add_geometry(pcl)
    vis.add_geometry(line_set)
    vis.add_geometry(block_pcl)
    return vis, pcl, line_set, block_pcl


def vis_config(vis):
    # Visualizer主要由view_control、render_option控制，需要定义其中的参数
    view_ctrl = vis.get_view_control()
    view_ctrl.set_front([0.12389456724560555, 0.044070795675724458, -0.99131624680297303])
    view_ctrl.set_lookat([-1.8012989955618823, -1.9890688531269081, 38.058145417087729])
    view_ctrl.set_up([0.99222458349823472, -0.017436165172234681, 0.12323293409572401])
    view_ctrl.set_zoom(0.35999999999999965)
    render = vis.get_render_option()
    render.point_size = 3
    # view_ctrl.change_field_of_view(0.6)
    # 更新渲染
    vis.poll_events()
    vis.update_renderer()


def point2pic(arr):
    intrinsic = np.asarray([
        [944.83371552882261, 0.0, 0.0],
        [0.0, 944.83371552882261, 0.0],
        [510.0, 545.0, 1.0],
        [0, 0, 0]
    ])
    extrinsic = np.asarray([
        [0.011853780358123728, -0.99222458349823484, -0.12389456724560555, 0.0],
        [0.99887624113930917, 0.017436165172234685, -0.044070795675724458, 0.0],
        [0.045888373022218418, -0.12323293409572403, 0.99131624680297303, 0.0],
        [0.2617594485982277, 2.9374055134243449, 1.5083058362627386, 1.0]
    ])
    points = np.ones((arr.shape[0], 4))
    points[:, 0:3] = arr[:, 0:3]
    pixel = np.matmul(np.matmul(intrinsic.T, extrinsic.T), points.T) / arr[:, 2]
    u, v, _ = pixel
    return u, v


def block_bbox(eval_data, block_index, line_set):
    """
    画出障碍物边框
    :param eval_data:
    :param block_index:
    :param line_set:
    :return:
    """
    if len(block_index) == 0:
        line_set.points = o3d.utility.Vector3dVector()
        line_set.lines = o3d.utility.Vector2iVector()
        return 0
    block_points = eval_data[block_index]
    x_min = block_points[:, 0].min()
    x_max = block_points[:, 0].max()
    y_min = block_points[:, 1].min()
    y_max = block_points[:, 1].max()
    z_min = block_points[:, 2].min()
    z_max = block_points[:, 2].max()

    points = [
        [x_min, y_min, z_min],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_min, y_min, z_max],
        [x_max, y_min, z_min],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
        [x_max, y_min, z_max],
    ]
    lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 0])
    return block_points


def win_init():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)  # 开启qt
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()  # 显示主窗口
    # MainWindow.raise_()  # 使窗口在最前面
    # print('load bg')
    vis1, eval_pcl, line_set, block_pcl = create_visualizer('eval', left=1210, top=50)
    print('create open3d')
    return app, ui, MainWindow, vis1, eval_pcl, line_set, block_pcl


def visual(eval_data, ui, vis1, eval_pcl, line_set, block_pcl, to_reset=True):
    start = time.time()
    img = QApplication.primaryScreen().grabWindow(0,1250, 150, 1920, 850)
    # 调用QtGui.QPixmap方法，打开一个图片，存放在变量png中
    # 在label里面，调用setPixmap命令，建立一个图像存放框，并将之前的图像png存放在这个框框里。
    ui.rail_view.setPixmap(QtGui.QPixmap(img))
    QtWidgets.qApp.processEvents()
    print('photo get: {}'.format(time.time()-start))
    start = time.time()
    index = np.where(eval_data[:, -1] == 3)[0]  # 障碍物index
    if len(index) >= 10:
        ui.flag_view.setPixmap(QtGui.QPixmap('ui/worning.jpeg'))
        ui.flag_view.setScaledContents(True)  # 自适应大小
        block_data = block_bbox(eval_data, index, line_set)
        distance = np.average(block_data[:, 2])  # 计算障碍物的距离
        ui.output.setText('距离障碍物:{:.2f}'.format(distance))
        QtWidgets.qApp.processEvents()  # 一边执行耗时程序，一边刷新界面的功能
    else:
        block_bbox(eval_data, index, line_set)
        ui.flag_view.setPixmap(QtGui.QPixmap('ui/run.png'))
        ui.flag_view.setScaledContents(True)
        ui.output.setText('未检测到障碍物')
        ui.bbox.setGeometry(QtCore.QRect(0, 0, 0, 0))
        QtWidgets.qApp.processEvents()  # 一边执行耗时程序，一边刷新界面的功能
    print('qt update: {}'.format(time.time()-start))
    start = time.time()
    eval_pcl.points = o3d.utility.Vector3dVector(eval_data[:, 0:3])
    eval_pcl.colors = o3d.utility.Vector3dVector(pc_colors(eval_data[:, -1]))
    vis1.update_geometry(eval_pcl)
    vis1.update_geometry(line_set)  # 更新geometry
    vis1.update_geometry(block_pcl)  # 更新geometry
    if to_reset:
        vis1.reset_view_point(True)

    vis_config(vis1)
    print('open3d update: {}'.format(time.time()-start))

def init(args):
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

    SemKITTI_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(sorted(list(SemKITTI_label_name.keys())))
    unique_label_str = [SemKITTI_label_name[x] for x in unique_label]  # 找出unique_label所对应的class

    my_model = model_builder.build(model_config)
    if os.path.exists(model_load_path):  # 读取check point
        checkpoint = torch.load(model_load_path)
        my_model.load_state_dict(checkpoint['model_state_dict'])
        print('load_checkpoint')

    my_model.to(pytorch_device)

    # 初始化Rail_sk类、初始化cylinder_dataset（附加数据增强） 并封装成dataloader
    _, val_dataset_loaders = rail_data_builder.build(dataset_config,
                                                     train_dataloader_config,
                                                     val_dataloader_config,
                                                     grid_size=grid_size, return_zyx=True)
    return my_model, val_dataset_loaders, val_folders, val_batch_size


def predict(args, queue: Queue):
    print('predict id: %d' % os.getpid())
    my_model, val_dataset_loaders, val_folders, val_batch_size = init(args)
    # evaluate
    my_model.eval()  # 测试
    hist_list = []
    with torch.no_grad():
        for i, val_dataset_loader in enumerate(val_dataset_loaders):
            if val_folders[i] != 'x':
                for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea, zyx) in enumerate(
                        val_dataset_loader):
                    start = time.time()
                    val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).cuda() for i in
                                      val_pt_fea]
                    val_grid_ten = [torch.from_numpy(i).cuda() for i in val_grid]
                    print('load-data:', time.time() - start)
                    start = time.time()
                    predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)
                    predict_labels = torch.argmax(predict_labels, dim=1)
                    predict_labels = predict_labels.cpu().detach().numpy()

                    for count, i_val_grid in enumerate(val_grid):
                        pre_point_label = predict_labels[
                            count, val_grid[count][:, 0], val_grid[count][:, 1], val_grid[count][:, 2]]
                        xyz = zyx[count]
                        xyz[:, [0, 2]] = xyz[:, [2, 0]]
                        evaluation = np.concatenate((xyz, pre_point_label[:, np.newaxis]), axis=1)
                        queue.put(evaluation)
                        print('predict_time:', time.time() - start)
        queue.put('exit')


def visual_open3d(queue: Queue):
    # print('visual id: %d' % os.getpid())
    evaluation = 0
    app, ui, MainWindow, vis1, eval_pcl, line_set, block_pcl = win_init()  # 初始化窗口
    # vis1, eval_pcl, line_set, block_pcl = create_visualizer('eval', left=1000, top=150)
    to_reset = True
    while True:
        if not queue.empty():
            evaluation = queue.get()
        if isinstance(evaluation, int):
            continue  # first in loop
        if isinstance(evaluation, str):
            break  # 收到结束信号则结束
        # print('start visual')
        # start = time.time()
        visual(evaluation, ui, vis1, eval_pcl, line_set, block_pcl, to_reset=to_reset)
        # eval_pcl.points = o3d.utility.Vector3dVector(evaluation[:, 0:3])
        # eval_pcl.colors = o3d.utility.Vector3dVector(pc_colors(evaluation[:, -1]))
        # vis1.update_geometry(eval_pcl)
        # vis1.update_geometry(line_set)  # 更新geometry
        # vis1.update_geometry(block_pcl)  # 更新geometry
        # if to_reset:
        #     vis1.reset_view_point(True)
        #     to_reset = False
        #
        # vis_config(vis1)
        # print('visual_time:', time.time() - start)

    vis1.destroy_window()
    # 退出函数，很奇怪，没有这个还不行
    # MainWindow.destroy()
    sys.exit(app.exec_())


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--config_path', default='config/rail.yaml')  # 读取体素分割大小、label、batchsize、验证集文件夹
    parser.add_argument('--model_load_path', default='model_save_dir/block_geo6_sparse.pt')
    args = parser.parse_args()
    print(' '.join(sys.argv))  # argv：用户输入的参数
    print(args)

    ctx = mp.get_context('spawn')
    print(ctx.get_start_method())
    print('main id: %d' % os.getpid())
    q = ctx.Queue()
    p1 = ctx.Process(target=predict, args=(args, q))
    p2 = ctx.Process(target=visual_open3d, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
