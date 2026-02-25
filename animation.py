import open3d as o3d
import os
import cv2
import numpy as np

def setup_render_scene(render, pcd):
    # 背景设为浅灰色，比较有高级感
    render.scene.set_background([0.9, 0.9, 0.9, 1.0])

    # 获取点云中心和跨度
    center = pcd.get_center()
    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    dist = np.max(extent) * 1.5  # 动态距离

    # 视角设置：从斜上方看下去 (俯视 45 度)
    eye = center + [dist, dist, dist]
    up = [0, 0, 1]  # 假设你的点云 Z 轴向上

    render.setup_camera(60.0, center, eye, up)

    # 材质设置（点的大小）
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.base_color = [0.2, 0.5, 0.9, 1.0] # 如果点云没颜色，给个统一颜色
    mtl.shader = "defaultUnlit"
    mtl.point_size = 3.0  # 点稍微大一点，看起来更扎实

    return mtl

def save_ply_sequence_to_video(folder_path, output_video="output.mp4"):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.ply')])

    # 初始化离线渲染器
    render = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)

    # 设置材质
    mtl = setup_render_scene(render, o3d.io.read_point_cloud(os.path.join(folder_path, files[0])))

    video_writer = None

    for i, file in enumerate(files):
        pcd = o3d.io.read_point_cloud(os.path.join(folder_path, file))

        # 第一次循环时初始化视频写入器
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_video, fourcc, 24, (1920, 1080))

        # 渲染这一帧
        render.scene.add_geometry(f"pcd_{i}", pcd, mtl)

        # 设置相机参数
        # 1. 计算点云中心
        center = pcd.get_center()

        # 2. 获取点云的大小（包围盒）
        extent = pcd.get_axis_aligned_bounding_box().get_extent()
        max_dim = max(extent)

        # 3. 设置相机位置 (Eye)
        # 我们让相机在 Z 轴上后退，距离根据模型大小自动调整
        # 这里的 2.0 是一个倍率，调大这个数，相机就离得更远
        eye = center + [0, 0, max_dim * 2.0]

        # 4. 应用到渲染器
        render.setup_camera(60.0, center, eye, [0, 1, 0])

        # 设置背景颜色（可选）
        render.scene.set_background([0.1, 0.1, 0.2, 1.0])

        # 调整摄像头视角（这里需要根据你的模型坐标调整）
        # 如果不知道坐标，可以先用 render.setup_camera(60, [0,0,0], [0,0,10], [0,1,0]) 试探
        render.setup_camera(60.0, [0, 0, 0], [0, 0, 5], [0, 1, 0])

        img = render.render_to_image()
        img_np = np.asarray(img)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        video_writer.write(img_bgr)
        render.scene.remove_geometry(f"pcd_{i}")
        print(f"正在处理第 {i+1}/{len(files)} 帧")

    video_writer.release()
    print(f"视频已保存至: {output_video}")

def save_rotating_ply_animation(folder_path, output_video="rotating_cloud.mp4"):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.ply')])
    render = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)

    # 1. 预处理：获取第一帧来确定场景中心和半径
    first_pcd = o3d.io.read_point_cloud(os.path.join(folder_path, files[0]))
    center = first_pcd.get_center()
    extent = first_pcd.get_axis_aligned_bounding_box().get_extent()

    # 设置相机的轨道半径（模型最大跨度的 2 倍比较稳妥）
    radius = np.max(extent) * 2.0
    # 设置相机的高度（让视角稍微俯视）
    camera_height = center[2] + (extent[2] * 0.5)

    # 视频写入初始化
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 24, (1920, 1080))

    # 材质设置
    mtl = o3d.visualization.rendering.MaterialRecord()
    mtl.shader = "defaultUnlit"
    mtl.point_size = 3.0

    total_frames = len(files)

    for i, file in enumerate(files):
        pcd = o3d.io.read_point_cloud(os.path.join(folder_path, file))

        # 2. 计算当前帧的角度 (一圈为 2*PI)
        # 这里的 1.0 表示在整个序列中刚好旋转一整圈
        angle = 2 * np.pi * (i / total_frames)

        # 3. 计算相机新的坐标 (Eye 位置)
        eye_x = center[0] + radius * np.cos(angle)
        eye_y = center[1] + radius * np.sin(angle)
        eye_z = camera_height

        # 4. 更新相机视角
        # setup_camera(fov, center, eye, up)
        render.setup_camera(60.0, center, [eye_x, eye_y, eye_z], [0, 0, 1])

        # 渲染并写入
        render.scene.add_geometry("frame_pcd", pcd, mtl)
        img = render.render_to_image()
        video_writer.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        render.scene.remove_geometry("frame_pcd")

        print(f"正在渲染旋转帧: {i+1}/{total_frames}")

    video_writer.release()

# 调用
# save_ply_sequence_to_video("./data")
save_rotating_ply_animation("./data")
