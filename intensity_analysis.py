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
    # 1. 定义输入和输出路径
    # input_dir = Path("./data/3")
    # input_dir = Path("./data/CP/CP-1525~1575/1")
    # input_dir = Path("./data/CP/CP-1525~1575/2")
    # input_dir = Path("./data/CP/CP-1525~1575/3")
    # input_dir = Path("./data/CP/comparision-LP-unpatterned-1550")
    # input_dir = Path("./data/LP/Gamma-X-patterned-1525~1575/1")
    # input_dir = Path("./data/20250118/1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure-better/CP/1/pos_mid-1.000~15.835-~42~90/phase-2500Exposure-1508~1528/c-1500exprosure")
    # input_dir = Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure-better\CP\1\pos_mid-1.000~15.835-~42~90\1480~1640-sweep\1-forw_back\forw")
    # input_dir = Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure\LP\1\sequence-pos_mid-1.000~15.835-~2~90-Gamma_M\forw_back\forw")
    # input_dir = Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure-better\CP\1\pos_mid-1.000~15.835-~42~90\phase_parttern-1500Exposure-bad_cross_polarized\unfiltered")
    # input_dir = Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure-better\CP\1\detail-pos_mid-1.000~15.835-~314~90-1508~1528")
    # input_dir = Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure-better\reference-better-sequenced\added")

    input_dirs = [
        # Path(r"D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\patterned\divided"),
        # Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure\CP\1\pos_mid-1.000(0.640~1.620)~15.835-42~90-1518"),
        # Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure\CP\1\pos_mid-1.000~15.835-~42~90\forw_back-added\forw"),
        # Path(r"D:\DELL\Documents\ExperimentDataToolkit\data\20250118\1480~1640-2cycle-sweep-back~forw-1.0Gain-5000Expsure\reference"),
        Path(r"D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640-old\patterned\divided"),
        # input_dir
    ]
    # input_dir = Path("./data/LP/Gamma-M-patterned-1525~1575/1")
    output_csv_path = Path("./rsl/optical_intensity_results.csv")

    # 2. 定义裁剪参数（圆形裁剪，固定中心）
    crop_shape_params = {
        'center_row': 0.46,   # 相对坐标
        'center_col': 0.47,   # 相对坐标 for phase pattern
        'radius': 0.35,       # 相对半径 bigger
        # 'radius': 0.31,       # 相对半径
        # 'radius': 0.15,       # 相对半径 smaller
        'inner_radius': 0,       # 相对半径
        'shape': 'circle',
        'relative': True     # 使用相对坐标
    }

    statistics_params = {
        # 'NA': [0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.42]
        'NA': [0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42]
    }

    # 3. 初始化分析器（可选添加标签）
    analyzer = OpticalIntensityAnalyzer(
        input_dirs=input_dirs,
        output_csv_path=output_csv_path,
        crop_shape_params=crop_shape_params,
        labels=None  # 或者传入 labels 字典，例如 labels=labels
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
