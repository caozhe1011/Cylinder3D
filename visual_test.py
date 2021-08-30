import numpy as np
import open3d as o3d


def pc_colors(arr):
    list = np.asarray([
        [127, 127, 127],
        [0, 255, 0],
        [127, 127, 127],
        [127, 127, 127],
        [127, 127, 127],
        [0, 0, 255],
    ])
    colors = []
    for i in arr:
        colors.append(list[i])

    return np.asarray(colors) / 255


def visual_3d(visual_flag: str, ori_path, eval_path):
    flag = visual_flag  # all表示显示所有错误点，rail表示只显示轨道错误点，pole表示只显示杆子错误点

    original = np.load(ori_path)
    label = original[:, -1].astype(np.int)
    ori_pcl = o3d.geometry.PointCloud()
    ori_pcl.points = o3d.utility.Vector3dVector(original[:, 0:3])
    ori_pcl.colors = o3d.utility.Vector3dVector(pc_colors(original[:, -1].astype(np.int)))

    evaluation = np.load(eval_path)
    predict_label = evaluation[:, -1].astype(np.int)
    eval_pcl = o3d.geometry.PointCloud()
    eval_pcl.points = o3d.utility.Vector3dVector(evaluation[:, 0:3])
    eval_pcl.colors = o3d.utility.Vector3dVector(pc_colors(evaluation[:, -1].astype(np.int)))
    # calculate Confusion matrix
    bin_count = np.bincount(3 * label + predict_label, minlength=3 ** 2)
    Confusion_matrix = bin_count.reshape(3, 3)
    recall = np.diag(Confusion_matrix) / np.sum(Confusion_matrix, axis=1)
    precision = np.diag(Confusion_matrix) / np.sum(Confusion_matrix, axis=0)
    iou = np.diag(Confusion_matrix) / (
            np.sum(Confusion_matrix, axis=0) + np.sum(Confusion_matrix, axis=1) - np.diag(Confusion_matrix))
    print("recall -", recall)
    print("precision -", precision)
    print("iou - ", iou)
    if flag == "all":
        error = np.where(label != predict_label)[0]
        np.asarray(ori_pcl.colors)[error] = [1, 0, 0]  # 所有预测错误的值
        print(error.shape[0] / original.shape[0])

    elif flag == "rail":  # 预测轨道错误结果，轨道预测为背景：红色，轨道预测为杆子:蓝色
        # 计算FP
        original_1 = np.where(original[:, -1] == 1)[0]  # 轨道label_index
        rail2back = np.where(evaluation[original_1][:, -1] == 0)[0]  # 从预测图中找出在原图中应该是轨道但是被预测为背景的值FP(预测错的)
        rail2back_idx = original_1[rail2back]

        rail2pole = np.where(evaluation[original_1][:, -1] == 2)[0]  # 从预测图中找出原图中应该是轨道但是被预测为杆子的值FP
        rail2pole_idx = original_1[rail2pole]
        if len(rail2back_idx) != 0:
            np.asarray(ori_pcl.colors)[rail2back_idx] = [1, 0, 0]  # rail2back 红色
        if len(rail2pole_idx) != 0:
            np.asarray(ori_pcl.colors)[rail2pole_idx] = [0, 0, 1]  # rail2pole 蓝色

        # 计算FN
        evaluation_1 = np.where(evaluation[:, -1] == 1)[0]  # 轨道预测值index
        rail2back = np.where(original[evaluation_1][:, -1] == 0)[0]
        rail2back_idx = evaluation_1[rail2back]

        rail2pole = np.where(original[evaluation_1][:, -1] == 2)[0]
        rail2pole_idx = evaluation_1[rail2pole]
        if len(rail2back_idx) != 0:
            np.asarray(ori_pcl.colors)[rail2back_idx] = [0.5, 0, 0]  # rail2back 暗红色
        if len(rail2pole_idx) != 0:
            np.asarray(ori_pcl.colors)[rail2pole_idx] = [0, 0, 0.5]  # rail2pole 深蓝色
    else:
        ori_pcl.colors = o3d.utility.Vector3dVector(pc_colors(original[:, -1].astype(np.int) + 3))
        original_2 = np.where(original[:, -1] == 2)[0]  # 杆子label_index
        pole2rail = np.where(evaluation[original_2][:, -1] == 1)[0]  # 从预测图中找出原图中应该是杆子但是被预测为轨道的值
        pole2rail_idx = original_2[pole2rail]

        pole2back = np.where(evaluation[original_2][:, -1] == 0)[0]  # 从预测图中找出原图中应该是杆子但是被预测为背景的值
        pole2back_idx = original_2[pole2back]
        if len(pole2rail_idx) != 0:
            np.asarray(ori_pcl.colors)[pole2rail_idx] = [1, 0, 0]  # pole2rail 红色
        if len(pole2back_idx) != 0:
            np.asarray(ori_pcl.colors)[pole2back_idx] = [0, 0, 1]  # pole2back 蓝色

        evaluation_2 = np.where(evaluation[:, -1] == 2)[0]  # 杆子预测值idex
        pole2rail = np.where(original[evaluation_2][:, -1] == 0)[0]
        pole2rail_idx = evaluation_2[pole2rail]

        pole2back = np.where(original[evaluation_2][:, -1] == 2)[0]
        pole2back_idx = evaluation_2[pole2back]
        if len(pole2rail_idx) != 0:
            np.asarray(ori_pcl.colors)[pole2rail_idx] = [0.5, 0, 0]  # pole2rail 暗红色
        if len(pole2back_idx) != 0:
            np.asarray(ori_pcl.colors)[pole2back_idx] = [0, 0, 0.5]  # pole2back 深蓝色

    o3d.visualization.draw_geometries([ori_pcl], width=1280, height=700,
                                      front=[0.88000862244702593, -0.46854402653354588, -0.077790228297773267],
                                      lookat=[-7.4168410301937158, 0.86326536790360087, 43.150439252735119],
                                      up=[0.47350048528007505, 0.85264177820285603, 0.22090560993918926],
                                      zoom=0.47999999999999976)
