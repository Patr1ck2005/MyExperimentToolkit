import os


def rename_png_files(folder_path, start, end):
    # 获取文件夹中所有的png文件
    png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

    # 如果文件夹中没有PNG文件
    if not png_files:
        print("没有找到PNG文件")
        return

    # 对文件进行排序，确保按文件名顺序
    png_files.sort()

    # 计算步长
    num_files = len(png_files)
    step = (end - start) / (num_files - 1) if num_files > 1 else 0

    # 重命名每个文件
    for idx, filename in enumerate(png_files):
        # 计算新的命名
        new_name = f"-{int(start + step * idx)}nm.png"

        # 获取原文件的完整路径
        old_file_path = os.path.join(folder_path, filename)

        # 获取新文件的完整路径
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"文件 {filename} 已重命名为 {new_name}")


# 使用示例
folder_path = r'D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure-better\CP\1\pos_mid-1.000~15.835-~42~90\phase-2500Exposure-1508~1528\reference-1500exposure'  # 请替换为实际路径
start = 1508  # 起始命名
end = 1528  # 终止命名

# 调用函数进行重命名
rename_png_files(folder_path, start, end)
