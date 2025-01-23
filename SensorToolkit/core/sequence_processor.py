from PIL import Image
import os
import numpy as np


class SequenceProcessor:
    def __init__(self, input_folder, window_size=3):
        self.input_folder = input_folder
        self.window_size = window_size
        self.output_folder = os.path.join(input_folder, 'window_average_rsl')

        # 如果 output 文件夹不存在，则创建它
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def apply_sliding_average(self, image_paths):
        # 读取图像并转换为numpy数组进行加和
        images = [np.array(Image.open(image_path), dtype=np.float32) for image_path in image_paths]

        # 计算图像的平均值
        averaged_image_np = np.mean(images, axis=0)

        # 转换回Image对象
        averaged_image = Image.fromarray(np.uint8(averaged_image_np))

        return averaged_image

    def process_images(self):
        # 获取输入文件夹中的所有图像文件
        filenames = sorted([f for f in os.listdir(self.input_folder) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        # 遍历图像文件，进行相邻图片叠加平均
        for i in range(len(filenames) - self.window_size + 1):
            image_paths = [os.path.join(self.input_folder, filenames[i + j]) for j in range(self.window_size)]

            # 处理多张相邻的图像
            print(f"处理图像: {', '.join(filenames[i:i + self.window_size])}")
            averaged_image = self.apply_sliding_average(image_paths)

            # 构建输出文件路径，添加 -slid_averaged 后缀
            output_filename = '-'.join(filenames[i:i + self.window_size]).rsplit('.', 1)[0] + '-slid_averaged.png'
            output_path = os.path.join(self.output_folder, output_filename)

            # 保存处理后的图像
            averaged_image.save(output_path)
            print(f"保存处理后的图像到: {output_path}")


class SequenceDivider:
    def __init__(self, input_folder1, input_folder2, output_folder, p2p: bool = False):
        self.input_folder1 = input_folder1
        self.input_folder2 = input_folder2
        self.output_folder = output_folder
        self.p2p = p2p

        # 如果 output 文件夹不存在，则创建它
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def divide_images(self, image1_path, image2_path):
        # 读取两张图像
        image1 = Image.open(image1_path)
        image2 = Image.open(image2_path)

        # 如果图像是RGBA模式（有透明通道），转换为RGB模式并处理透明区域
        if image1.mode == 'RGBA':
            alpha1 = np.array(image1.split()[3], dtype=np.float32)  # 获取透明度通道
            image1 = image1.convert('RGB')  # 仅保留RGB部分
        else:
            alpha1 = np.ones((image1.height, image1.width), dtype=np.float32)  # 没有透明通道，使用全1

        if image2.mode == 'RGBA':
            alpha2 = np.array(image2.split()[3], dtype=np.float32)  # 获取透明度通道
            image2 = image2.convert('RGB')  # 仅保留RGB部分
        else:
            alpha2 = np.ones((image2.height, image2.width), dtype=np.float32)  # 没有透明通道，使用全1

        # 转换为numpy数组进行处理
        image1_np = np.array(image1, dtype=np.float32)
        image2_np = np.array(image2, dtype=np.float32)

        # 计数非透明像素点
        non_zero_count1 = np.count_nonzero(alpha1)
        non_zero_count2 = np.count_nonzero(alpha2)
        # 统计非透明像素点比例
        non_zero_ratio1 = non_zero_count1 / (image1.height * image1.width)
        non_zero_ratio2 = non_zero_count2 / (image2.height * image2.width)
        # 将透明区域（alpha为0的地方）处理成0
        image1_np[alpha1 == 0] = 0
        image2_np[alpha2 == 0] = 0

        # 防止除以零，避免出现无穷大或无效值
        image2_np[image2_np == 0] = 1e-6  # 将值为0的像素设置为非常小的数值

        # 执行图像相除
        if self.p2p:
            result_np = np.clip(255 * image1_np / image2_np, 0, 255)  # 保证结果在0到255范围内
        else:
            result_np = np.clip(255 * image1_np / image2_np.mean()*non_zero_ratio2, 0, 255)  # 保证结果在0到255范围内

        # 转换回Image对象，确保结果是RGB模式
        result_image = Image.fromarray(np.uint8(result_np))

        return result_image

    def process_images(self):
        # 获取第一个文件夹中的所有图像文件
        filenames1 = sorted([f for f in os.listdir(self.input_folder1) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        filenames2 = sorted([f for f in os.listdir(self.input_folder2) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        # 遍历文件名，确保两个文件夹中的图像一一对应
        for filename1, filename2 in zip(filenames1, filenames2):
            image1_path = os.path.join(self.input_folder1, filename1)
            image2_path = os.path.join(self.input_folder2, filename2)

            # 处理每对对应的图像
            print(f"处理图像: {filename1} 和 {filename2}")
            result_image = self.divide_images(image1_path, image2_path)

            # 构建输出文件路径
            output_filename = filename1.rsplit('.', 1)[0] + '-result.png'
            output_path = os.path.join(self.output_folder, output_filename)

            # 保存处理后的图像
            result_image.save(output_path)
            print(f"保存处理后的图像到: {output_path}")

