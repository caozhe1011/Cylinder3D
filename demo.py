import os
import numpy as np
import open3d as o3d


def pcd_color(ndarray):
    color = []
    ndarray = ndarray.astype(np.int)
    color_list = [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
    ]
    for i in ndarray:
        color.append(color_list[i])
    return np.asarray(color)


o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
files = os.listdir(r"E:\0dataset\rail\labeled_rail\pc_npy\a3")

vis = o3d.visualization.Visualizer()
# 创建播放窗口
vis.create_window('rail')
pointcloud = o3d.geometry.PointCloud()
to_reset = True
vis.add_geometry(pointcloud)

for f in files:
    pcd = np.load(r'E:\0dataset\rail\labeled_rail\pc_npy\a3\\' + f)
    pointcloud.points = o3d.utility.Vector3dVector(pcd[:, 0:3])
    pointcloud.colors = o3d.utility.Vector3dVector(pcd_color(pcd[:, -1]))
    vis.update_geometry(pointcloud)  # 更新geometry
    if to_reset:
        vis.reset_view_point(True)
        to_reset = False
    # Visualizer主要由view_control、render_option控制，需要定义其中的参数
    view_ctrl = vis.get_view_control()
    view_ctrl.set_front([0.12389456724560555, 0.044070795675724458, -0.99131624680297303])
    view_ctrl.set_lookat([-1.8012989955618823, -1.9890688531269081, 38.058145417087729])
    view_ctrl.set_up([0.99222458349823472, -0.017436165172234681, 0.12323293409572401])
    view_ctrl.set_zoom(0.35999999999999965)
    render = vis.get_render_option()
    render.point_size = 3
    # 更新渲染
    vis.poll_events()
    vis.update_renderer()
vis.destroy_window()
