# 导入类
from SensorToolkit.core.sequence_processor import SequenceDivider

# 设置文件夹路径
input_folder1 = r'D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\0.42\window_average_rsl'  # 第一个文件夹路径
input_folder2 = r'D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\unpatterned\window_average_rsl'  # 第二个文件夹路径
output_folder = r'D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\0.42\divided'  # 输出文件夹路径

divider = SequenceDivider(input_folder1, input_folder2, output_folder, p2p=False)

# 执行图像序列相除
divider.process_images()
