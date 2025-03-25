# process_optical_intensity.py

from pathlib import Path
import logging

import numpy as np

from SensorToolkit import OpticalIntensityAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 导入标签（可选）
# from labels import labels


def main():
    input_dirs = [
        Path(r"D:\DELL\Documents\S4Simulation\img\1500~1600\npy\conversion_efficiency\C4"),  # temp
    ]
    # input_dir = Path("./data/LP/Gamma-M-patterned-1525~1575/1")
    output_csv_path = Path("./rsl/optical_intensity_results.csv")

    # 2. 定义裁剪参数（圆形裁剪，固定中心）
    crop_shape_params = {
        'center_col': 0.5,
        'center_row': 0.5,
        'radius': 0.5,
        'inner_radius': 0,
        'shape': 'circle',
        'relative': True     # 使用相对坐标
    }

    statistics_params = {
        # 'NA': [0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.42]
        'NA': [0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42],
        # 'hollow_proportion': 0.9,
        # 'average_weight': 'even',
        'average_weight': 'gaussian',
    }

    # 3. 初始化分析器（可选添加标签）
    analyzer = OpticalIntensityAnalyzer(
        input_dirs=input_dirs,
        output_csv_path=output_csv_path,
        crop_shape_params=crop_shape_params,
        labels=None,  # 或者传入 labels 字典，例如 labels=labels
        file_extension='.npy',
    )

    # 4. 添加标签（如果有）
    # analyzer.add_label("1550.bmp", "Sample_A")
    # analyzer.add_label("1600.bmp", "Sample_B")
    # 或者在初始化时传入 labels 字典

    # 5. 执行批量处理
    # analyzer.process_all()
    analyzer.statistic_all(statistics_params=statistics_params,)

    # 6. 打印处理结果（可选）
    # print(analyzer.results)


if __name__ == "__main__":
    main()
