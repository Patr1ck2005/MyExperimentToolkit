import os

def rename_image_files_to_sequence(folder_path):
    # 获取文件夹中所有的图片文件（支持 .jpg, .jpeg, .png）
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 如果文件夹中没有图片文件
    if not image_files:
        print("没有找到图片文件")
        return

    # 对文件进行排序，确保按文件名顺序
    image_files.sort()

    # 重命名每个文件为图片序列
    for idx, filename in enumerate(image_files, start=0):
        # 获取文件扩展名
        file_extension = os.path.splitext(filename)[1]

        # 计算新的命名
        new_name = f"image_{idx}{file_extension}"

        # 获取原文件的完整路径
        old_file_path = os.path.join(folder_path, filename)

        # 获取新文件的完整路径
        new_file_path = os.path.join(folder_path, new_name)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"文件 {filename} 已重命名为 {new_name}")

# 使用示例
folder_path = r'D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure\LP\1\sequence-pos_mid-1.000~15.835-~2~90-Gamma_X'  # 请替换为实际路径

# 调用函数进行重命名
rename_image_files_to_sequence(folder_path)
