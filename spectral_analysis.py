from pathlib import Path
from SensorToolkit.spectral_analysis import SpectralAnalyzer


def main():
    # 定义裁剪后图像所在的目录
    images_directory = Path("./temp")

    # 初始化光谱分析器
    spectral_analyzer = SpectralAnalyzer(
        images_dir=images_directory,
        boundary_na=0.42,
        wavelength_order='descending'  # 或 'ascending'
    )

    # 运行3D光谱可视化
    spectral_analyzer.run_3d_volume_visualization_pyvista(output_path=Path("./spectral_3d_visualization.png"))


if __name__ == "__main__":
    main()
