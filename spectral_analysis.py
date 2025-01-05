import logging
from pathlib import Path
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

    # 定义裁剪后图像所在的目录
    data_directory = Path("./temp")

    # 初始化光谱分析器
    spectral_analyzer = SpectralAnalyzer(
        working_dir=data_directory,
        boundary_na=0.42,
        wavelength_order='descending',  # 或 'ascending'
        file_type='npy'  # 'npy' 或 'png'
    )

    # 定义颜色范围（可选）
    color_range = (0.0, 1.0)  # 根据需要调整
    # color_range = None  # 根据需要调整

    # 运行3D光谱可视化，并保存为PNG和HTML文件
    spectral_analyzer.load_data_files()
    (spectral_analyzer
    .apply_2d_upsample(
        zoom_factor=5.0,
        order=2
    ))
    spectral_analyzer.visualize_3d_volume_pyvista(
        output_path=Path("./spectral_3d_visualization.png"),
        html_path=Path("./spectral_3d_visualization.html"),
        clim=color_range
    )

    # # 运行3D光谱可视化，并保存为PNG和HTML文件
    # spectral_analyzer.load_images()
    # spectral_analyzer.visualize_two_planes_intensity_map(
    #     angles=[30, 120+45],
    #     output_path=Path("./two_planes_intensity_map.png")
    # )

if __name__ == "__main__":
    main()
