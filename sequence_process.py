from SensorToolkit.core.sequence_processor import SequenceProcessor

# 使用上传的文件路径来处理图像
input_folder = r'D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure\LP\1\sequence-pos_mid-1.000~15.835-~2~90-Gamma_M\forw_back\forw'  # 输入文件夹路径
processor = SequenceProcessor(input_folder, window_size=3)
processor.process_images()
