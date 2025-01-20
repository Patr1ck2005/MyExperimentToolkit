from SensorToolkit.core.sequence_processor import SequenceProcessor

# 使用上传的文件路径来处理图像
input_folder = r'D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\unpatterned'  # 输入文件夹路径
processor = SequenceProcessor(input_folder, window_size=5)
processor.process_images()
