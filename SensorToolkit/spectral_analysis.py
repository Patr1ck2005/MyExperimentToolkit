# core/spectral_analysis.py

from pathlib import Path
import logging
import pyvista as pv
import numpy as np
from PIL import Image
import pandas as pd
from scipy.interpolate import interp1d
from SensorToolkit.utils.filename_parser import FilenameParser  # 确认导入路径


class SpectralAnalyzer:
    """
    基于裁剪后的光强分布图进行光谱分析的类。
    提供多种光谱预案，如3D可视化等。
    """

    def __init__(
            self,
            images_dir: Path,
            boundary_na: float = 0.42,
            wavelength_order: str = 'descending',  # 'ascending' 或 'descending'
            filename_delimiter: str = '-'  # 新增参数
    ):
        """
        初始化光谱分析器。

        :param images_dir: 裁剪后保存的 PNG 图像所在的目录。
        :param boundary_na: 动量空间成像的数值孔径（NA）边界。
        :param wavelength_order: 波长排序方式，'ascending' 或 'descending'。
        :param filename_delimiter: 文件名中用于分割不同部分的分隔符。
        """
        self.images_dir = images_dir
        self.boundary_na = boundary_na
        self.wavelength_order = wavelength_order
        self.image_data = {}
        self.wavelengths = []
        self.filename_parser = FilenameParser(delimiter=filename_delimiter)  # 使用 FilenameParser
        logging.info(f"初始化 SpectralAnalyzer，图像目录: {self.images_dir}, 边界 NA: {self.boundary_na}")

    def load_images(self):
        """
        加载图像数据并提取波长信息。
        假设图像文件名中包含波长信息，例如 "1550.0-label1-label2.png"。
        仅加载非透明部分的图像数据。
        """
        image_files = list(self.images_dir.glob("*.png"))  # 如果仅处理裁剪后的PNG，考虑使用 "*_cropped.png"
        if not image_files:
            logging.warning(f"在 {self.images_dir} 中未找到任何处理后的 PNG 文件。")
            return

        logging.info(f"加载 {len(image_files)} 个图像文件。")
        for image_file in image_files:
            filename = image_file.name
            # 使用 FilenameParser 提取波长
            info = self.filename_parser.extract_info(filename)
            wavelength = info.get('wavelength_nm')
            if wavelength is None:
                logging.warning(f"文件 {filename} 没有波长信息，跳过。")
                continue
            self.wavelengths.append(wavelength)
            # 使用 PIL 加载图像以处理透明度
            try:
                with Image.open(image_file) as img:
                    img = img.convert("RGBA")  # 确保有 alpha 通道
                    data = np.array(img)
                    # 创建透明度掩码（alpha > 0）
                    alpha_mask = data[:, :, 3] > 0
                    # 转换为灰度图像
                    gray_image = np.array(img.convert("L"))
                    # 仅保留非透明部分
                    gray_image[~alpha_mask] = 0
                    self.image_data[wavelength] = gray_image
                logging.info(f"加载波长 {wavelength} nm 的图像: {filename}")
            except Exception as e:
                logging.error(f"加载图像 {filename} 时发生错误: {e}")
                continue

        # 排序波长
        self.wavelengths = sorted(self.wavelengths, reverse=(self.wavelength_order == 'descending'))
        logging.info(f"波长排序 ({self.wavelength_order}): {self.wavelengths}")

    def resample_spectral_stack(self, spectral_stack, original_z, target_z):
        """
        将 spectral_stack 从 original_z 重新采样到 target_z。

        :param spectral_stack: 原始的3D数组 (rows, cols, depth)
        :param original_z: 原始的 Z 轴坐标（非均匀）
        :param target_z: 目标的 Z 轴坐标（均匀）
        :return: 重新采样后的 spectral_stack
        """
        rows, cols, depth = spectral_stack.shape
        resampled_stack = np.zeros((rows, cols, len(target_z)))

        for i in range(rows):
            for j in range(cols):
                # 获取当前像素点的光谱数据
                intensity = spectral_stack[i, j, :]
                # 创建插值函数
                interp_func = interp1d(original_z, intensity, bounds_error=False, fill_value=0)
                # 插值到目标 Z 轴
                resampled_stack[i, j, :] = interp_func(target_z)

        return resampled_stack

    def visualize_3d_volume_pyvista(self, output_path: Path = Path("./spectral_3d_volume_pyvista.png")):
        """
        使用 PyVista 生成3D体积渲染可视化。

        :param output_path: 可视化图像的保存路径（PNG文件）。
        """
        if not self.image_data:
            logging.warning("没有加载任何图像数据，无法进行3D可视化。")
            return

        # 假设所有图像的尺寸相同
        sample_wavelength = self.wavelengths[0]
        image_shape = self.image_data[sample_wavelength].shape
        rows, cols = image_shape
        original_z = np.array(self.wavelengths)  # 原始 Z 轴（非均匀）

        # 定义新的均匀 Z 轴
        num_slices = 30  # 定义目标切片数量，根据需要调整
        target_z = np.linspace(original_z.min(), original_z.max(), num_slices)

        # 创建3D数据体
        spectral_stack = np.zeros((rows, cols, len(original_z)))

        for idx, wavelength in enumerate(self.wavelengths):
            spectral_stack[:, :, idx] = self.image_data[wavelength]

        # 重新采样 spectral_stack 到均匀 Z 轴
        logging.info("开始重新采样 spectral_stack 到均匀 Z 轴...")
        resampled_stack = self.resample_spectral_stack(spectral_stack, original_z, target_z)
        logging.info("重新采样完成。")

        # 归一化数据
        max_intensity = np.max(resampled_stack)
        if max_intensity > 0:
            resampled_stack = resampled_stack / max_intensity

        # 创建 PyVista 的 ImageData
        grid = pv.ImageData()
        grid.dimensions = resampled_stack.shape
        grid.spacing = (1, 1, (target_z[1] - target_z[0])*10)  # 假设 X 和 Y 间距为1，Z 间距为均匀
        grid.origin = (0, 0, target_z[0])

        # 将数据添加为 'intensity' 标量
        grid.point_data["intensity"] = resampled_stack.flatten(order="F")  # 使用 point_data 代替 point_arrays

        # 创建绘图器
        plotter = pv.Plotter()

        # 添加体积渲染
        opacity = [0, 0.1, 1.0]  # 自定义透明度映射
        cmap = "viridis"

        logging.info("开始添加体积渲染到绘图器...")
        plotter.add_volume(grid, scalars="intensity", cmap=cmap, opacity=opacity, shade=True)
        logging.info("体积渲染添加完成。")

        # # 设置坐标轴标签
        # plotter.set_xlabel("X 像素")
        # plotter.set_ylabel("Y 像素")
        # plotter.set_zlabel("波长 (nm)")

        # 添加标题
        plotter.add_title("3D 体积渲染光谱可视化")

        # 渲染并保存为PNG
        logging.info(f"开始渲染并保存截图至 {output_path}...")
        plotter.show(screenshot=str(output_path))
        plotter.close()
        logging.info(f"3D 体积渲染光谱可视化已保存至: {output_path}")

    def run_3d_volume_visualization_pyvista(self, output_path: Path = Path("./spectral_3d_volume_pyvista.png")):
        """
        执行3D体积渲染光谱可视化的完整流程。

        :param output_path: 可视化图像的保存路径。
        """
        self.load_images()
        self.visualize_3d_volume_pyvista(output_path=output_path)

