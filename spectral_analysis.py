import logging
from pathlib import Path

import numpy as np

from SensorToolkit.spectral_analyzer import SpectralAnalyzer

def main():
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.StreamHandler()  # 将日志输出到控制台
        ]
    )


    # 定义颜色范围（可选）
    # color_range = (0.0, 1.0)  # 根据需要调整
    color_range = None  # 根据需要调整

    # opacity = [0, 0.1, 0.5, 0.75, 1.0]  # 自定义透明度映射
    opacity = 'linear'  # 定义透明度映射
    # opacity = [int(val > 0) for val in np.linspace(-0.01, 1, 255)]  # 定义透明度映射
    # cmap = "viridis"
    # cmap = "hot"
    cmap = "magma"

    if False:
        # 定义裁剪后图像所在的目录
        # data_directory = Path("./temp/A48-25deg-1530~1570-lossless-Au")
        data_directory = Path("./temp/A48-25deg-1530~1670")
        # data_directory = Path("./temp/A48-25deg-R1.05-1500~1570")
        # data_directory = Path("./temp/A48-25deg-R1.10-1500~1570")
        # data_directory = Path("./temp/A128-20deg-1545~1555")

        # 初始化光谱分析器
        spectral_analyzer = SpectralAnalyzer(
            working_dir=data_directory,
            boundary_na=0.42,
            wavelength_order='descending',  # 或 'ascending'
            file_type='npy',  # 'npy' 或 'png'
        )

        # 运行3D光谱可视化，并保存为PNG和HTML文件
        spectral_analyzer.load_data_files()
        (spectral_analyzer
        .apply_2d_upsample(
            zoom_factor=5.0,
            order=3
        )
        .crop_data(
            center_row=0.5,   # 相对坐标
            center_col=0.5,   # 相对坐标
            radius=0.5,       # 相对半径
            inner_radius=0,       # 相对半径
            shape='circle',
            relative=True     # 使用相对坐标
        )
        )
        spectral_analyzer.visualize_3d_volume_pyvista(
            output_path=Path("./spectral_3d_visualization.png"),
            html_path=Path("./spectral_3d_visualization.html"),
            opacity=opacity,
            cmap=cmap,
            clim=color_range,
            angles=[0, 135],
        )
        spectral_analyzer.visualize_two_planes_intensity_map(
            angles=[0, 135],
            output_path=Path("./sim_two_planes_intensity_map.png"),
            vmax=1,
        )

    # ------------------------------------------------------------------------------------------------------------------

    if True:
        # 定义裁剪后图像所在的目录
        data_directory = Path(r"D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\divided")
        # data_directory = Path(r"D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\window_average_rsl\5")
        # data_directory = Path(r"D:\DELL\Documents\ExperimentDataToolkit\temp\1480~1640\unpatterned\window_average_rsl")
        # data_directory = Path("./temp/1-filtered")
        # data_directory = Path("./temp/3-filtered")
        # data_directory = Path("./temp/Gamma-M")
        # data_directory = Path("./temp/Gamma-X")

        # 初始化光谱分析器
        spectral_analyzer = SpectralAnalyzer(
            working_dir=data_directory,
            boundary_na=0.42,
            wavelength_order='descending',  # 或 'ascending'
            file_type='png',  # 'npy' 或 'png'
            max_wavelength=1601
        )

        # 定义颜色范围（可选）
        # color_range = (0.0, 1.0)  # 根据需要调整
        color_range = None  # 根据需要调整

        start_angle = +0

        # 运行3D光谱可视化，并保存为PNG和HTML文件
        spectral_analyzer.load_data_files()
        # spectral_analyzer.trans_to_efficiency()
        spectral_analyzer.visualize_3d_volume_pyvista(
            output_path=Path("./spectral_3d_visualization.png"),
            html_path=Path("./spectral_3d_visualization.html"),
            opacity=opacity,
            cmap=cmap,
            clim=color_range,
            angles=[start_angle, start_angle+145],
        )

        # 运行3D光谱可视化，并保存为PNG和HTML文件
        spectral_analyzer.visualize_two_planes_intensity_map(
            angles=[start_angle, start_angle+145],
            output_path=Path("./two_planes_intensity_map.png"),
            cmap=cmap,
        )

if __name__ == "__main__":
    main()
