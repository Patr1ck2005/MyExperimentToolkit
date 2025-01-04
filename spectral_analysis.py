import logging
from pathlib import Path
from SensorToolkit.spectral_analysis import SpectralAnalyzer

def main():
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别为 INFO
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 设置日志格式
        handlers=[
            logging.StreamHandler()  # 将日志输出到控制台
        ]
    )

    # 定义裁剪后图像所在的目录
    images_directory = Path("./temp")

    # 初始化光谱分析器
    spectral_analyzer = SpectralAnalyzer(
        images_dir=images_directory,
        boundary_na=0.42,
        wavelength_order='descending'  # 或 'ascending'
    )

    # 定义颜色范围（可选）
    color_range = (0.0, 0.5)  # 根据需要调整
    color_range = None  # 根据需要调整

    # # 运行3D光谱可视化，并保存为PNG和HTML文件
    # spectral_analyzer.run_3d_volume_visualization_pyvista(
    #     output_path=Path("./spectral_3d_visualization.png"),
    #     html_path=Path("./spectral_3d_visualization.html"),
    #     clim=color_range  # 或者设置为 None 使用自动计算
    # )

    # 运行3D光谱可视化，并保存为PNG和HTML文件
    spectral_analyzer.run_two_planes_intensity_map_visualization(angles=(0, 90))

if __name__ == "__main__":
    main()
