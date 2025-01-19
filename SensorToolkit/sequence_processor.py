from PIL import Image, ImageFilter
import os


class SequenceProcessor:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.output_folder = os.path.join(input_folder, 'window_average_rsl')

        # 如果 output 文件夹不存在，则创建它
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def apply_window_average(self, image_path, window_size=5):
        # 读取图像
        image = Image.open(image_path)

        # 执行窗口平均化
        processed_image = image.filter(ImageFilter.BoxBlur(window_size))

        return processed_image

    def process_images(self):
        # 获取输入文件夹中的所有图像文件
        for filename in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, filename)

            if os.path.isfile(file_path) and filename.lower().endswith(('png', 'jpg', 'jpeg')):
                # 处理每个图像文件
                print(f"处理图像: {filename}")
                processed_image = self.apply_window_average(file_path)

                # 构建输出文件路径
                output_path = os.path.join(self.output_folder, filename)
                processed_image.save(output_path)
                print(f"保存处理后的图像到: {output_path}")

