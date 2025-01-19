from SensorToolkit.sequence_processor import SequenceProcessor

# 使用上传的文件路径来处理图像
input_folder = '/mnt/data'  # 输入文件夹路径
processor = SequenceProcessor(input_folder)
processor.process_images()
