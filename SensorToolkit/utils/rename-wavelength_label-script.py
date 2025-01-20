import os

def rename_files_by_creation_time_with_range(folder_path, start, end):
    # 获取文件夹中所有的图片文件（支持 .jpg, .jpeg, .png）
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 如果文件夹中没有文件
    if not files:
        print("没有找到图片文件")
        return

    # 获取每个文件的修改时间，并按创建时间排序
    files_with_time = [(f, os.path.getmtime(os.path.join(folder_path, f))) for f in files]
    files_sorted = sorted(files_with_time, key=lambda x: x[1])  # 按时间排序

    # 计算步长
    num_files = len(files_sorted)
    step = (end - start) / (num_files - 1) if num_files > 1 else 0

    # 重命名文件
    for idx, (filename, _) in enumerate(files_sorted):
        # 计算新的命名
        new_name = f"-{(start + step * idx):.2f}nm{os.path.splitext(filename)[1]}"

        # 获取原文件的完整路径
        old_file_path = os.path.join(folder_path, filename)

        # 获取新文件的完整路径
        new_file_path = os.path.join(folder_path, new_name)

        # 检查目标文件是否已存在，避免覆盖
        if os.path.exists(new_file_path):
            print(f"文件 {new_name} 已存在，跳过重命名.")
            continue

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"文件 {filename} 已重命名为 {new_name}")

# 使用示例
folder_path = r'D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure\LP\1\sequence-pos_mid-1.000~15.835-~2~90-Gamma_X\back'  # 请替换为实际路径
# start = 1480  # 起始命名
# end = 1640  # 终止命名
start = 1640  # 起始命名
end = 1480  # 终止命名
# start = 1508  # 起始命名
# end = 1528  # 终止命名

# 调用函数进行重命名
rename_files_by_creation_time_with_range(folder_path, start, end)
