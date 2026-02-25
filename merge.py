import open3d as o3d
import os

def merge_ply_files(input_folder, output_filename):
    combined_pcd = o3d.geometry.PointCloud()

    # 获取文件夹内所有 ply 文件
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.ply')])

    for file in files:
        file_path = os.path.join(input_folder, file)
        pcd = o3d.io.read_point_cloud(file_path)
        combined_pcd += pcd  # 直接相加合并点云
        print(f"已添加: {file}")

    # 保存合并后的文件
    o3d.io.write_point_cloud(output_filename, combined_pcd)
    print(f"合并完成！保存为: {output_filename}")

# 使用
merge_ply_files("./data", "merged_scene.ply")
