from SensorToolkit.core.sequence_processor import SequenceProcessor

# 使用上传的文件路径来处理图像
input_folder = r'D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\Gamma_X'  # 输入文件夹路径
processor = SequenceProcessor(input_folder, window_size=3)
processor.process_images()
