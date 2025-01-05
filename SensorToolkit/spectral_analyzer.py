# spectral_analysis.py

from pathlib import Path
import logging

import pandas as pd
import pyvista as pv
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d

from SensorToolkit.core.data_processing import DataProcessor3D, DataProcessor
from SensorToolkit.utils.filename_parser import FilenameParser  # 确认导入路径
import matplotlib.pyplot as plt  # 新增导入
from scipy.ndimage import map_coordinates, gaussian_filter  # 用于插值


class SpectralAnalyzer:
    def __init__(
            self,
            working_dir: Path,
            boundary_na: float = 0.42,
            wavelength_order: str = 'descending',  # 'ascending' 或 'descending'
            filename_delimiter: str = '-',  # 新增参数
            file_type: str = 'png'  # 新增参数，支持 'png' 或 'npy'
    ):
        """
        初始化光谱分析器。

        :param working_dir: 待处理的数据文件所在的目录。
        :param boundary_na: 动量空间成像的数值孔径（NA）边界。
        :param wavelength_order: 波长排序方式，'ascending' 或 'descending'。
        :param filename_delimiter: 文件名中用于分割不同部分的分隔符。
        :param file_type: 文件类型，支持 'png' 或 'npy'。
        """
        self.working_dir = working_dir
        self.boundary_na = boundary_na
        self.wavelength_order = wavelength_order
        self.file_type = file_type.lower()
        self.load_data = {}
        self.wavelengths = []
        self.comparison_data = None  # (可选) 对照组数据
        self.comparison_data_path = working_dir / "comparison-Au-processed.png"  # (可选) 对照组数据
        self.filename_parser = FilenameParser(delimiter=filename_delimiter)  # 使用 FilenameParser
        logging.info(f"初始化 SpectralAnalyzer，工作目录: {self.working_dir}, 边界 NA: {self.boundary_na}")

    def load_numpy_arrays(self):
        """
        加载 .npy 文件数据并提取波长信息。
        假设文件名中包含波长信息，例如 "1550.0-label1-label2.npy"。
        """
        npy_files = list(self.working_dir.glob("*.npy"))
        if not npy_files:
            logging.warning(f"在 {self.working_dir} 中未找到任何 .npy 文件。")
            return

        logging.info(f"加载 {len(npy_files)} 个 .npy 文件。")
        for npy_file in npy_files:
            filename = npy_file.name
            # 使用 FilenameParser 提取波长
            info = self.filename_parser.extract_info(filename)
            wavelength = info.get('wavelength_nm')
            if wavelength is None:
                logging.warning(f"文件 {filename} 没有波长信息，跳过。")
                continue

            self.wavelengths.append(wavelength)
            # 加载 .npy 文件
            try:
                data = np.load(npy_file)
                if data.ndim != 2:
                    logging.warning(f"文件 {filename} 的数据维度为 {data.ndim}，期望为 2D，跳过。")
                    continue
                self.load_data[wavelength] = data
                logging.info(f"加载波长 {wavelength} nm 的数据: {filename}")
            except Exception as e:
                logging.error(f"加载 .npy 文件 {filename} 时发生错误: {e}")
                continue

        # 排序波长
        self.wavelengths = sorted(self.wavelengths, reverse=(self.wavelength_order == 'descending'))
        logging.info(f"波长排序 ({self.wavelength_order}): {self.wavelengths}")

    def load_images(self):
        """
        加载 PNG 图像数据并提取波长信息。
        假设图像文件名中包含波长信息，例如 "1550.0-label1-label2.png"。
        """
        image_files = list(self.working_dir.glob("*.png"))
        if not image_files:
            logging.warning(f"在 {self.working_dir} 中未找到任何处理后的 PNG 文件。")
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
                    gray_image = np.array(img.convert("L")).astype(np.float32)
                    # 仅保留非透明部分
                    gray_image[~alpha_mask] = -1
                    self.load_data[wavelength] = gray_image / 255  # 归一化数据
                logging.info(f"加载波长 {wavelength} nm 的图像: {filename}")
            except Exception as e:
                logging.error(f"加载图像 {filename} 时发生错误: {e}")
                continue

        # 排序波长
        self.wavelengths = sorted(self.wavelengths, reverse=(self.wavelength_order == 'descending'))
        logging.info(f"波长排序 ({self.wavelength_order}): {self.wavelengths}")

    def load_data_files(self):
        """
        根据文件类型加载数据。
        """
        if self.file_type == 'png':
            self.load_images()
        elif self.file_type == 'npy':
            self.load_numpy_arrays()
        else:
            logging.error(f"不支持的文件类型: {self.file_type}")

    def trans_to_efficiency(self):
        # 使用 PIL 加载图像以处理透明度
        with Image.open(self.comparison_data_path) as img:
            img = img.convert("RGBA")  # 确保有 alpha 通道
            data = np.array(img)
            # 创建透明度掩码（alpha > 0）
            alpha_mask = data[:, :, 3] > 0
            # 转换为灰度图像
            gray_image = np.array(img.convert("L")).astype(np.float32)
            # 仅保留非透明部分
            gray_image[~alpha_mask] = -1
            self.comparison_data = gray_image / 255  # 归一化数据
        logging.info(f"对照图像: {self.comparison_data_path}")
        for wavelength, image in self.load_data.items():
            image[alpha_mask] /= self.comparison_data[alpha_mask]
            image = np.clip(image, 0, 1)
            self.load_data[wavelength] = image

    def slice_cylinder_quarter(self, spectral_stack, center=None, radius=None, angle_range=(0, 90)):
        """
        对圆柱型数据进行切片处理，切掉 1/4 圆弧区域。

        :param spectral_stack: 3D 数据体 (rows, cols, depth)。
        :param center: 圆心的坐标 (x, y)。默认为图像中心。
        :param radius: 圆的半径。默认为数据的最大对角线距离。
        :param angle_range: 切片的角度范围（以度为单位），默认切掉 1/4 圆弧 (0, 90)。
        :return: 切片处理后的 spectral_stack。
        """
        rows, cols, depth = spectral_stack.shape

        # 默认圆心为图像中心
        if center is None:
            center = (cols / 2, rows / 2)

        # 默认半径为图像对角线的一半
        if radius is None:
            radius = np.sqrt((cols / 2) ** 2 + (rows / 2) ** 2)

        x_center, y_center = center
        angle_min, angle_max = np.deg2rad(angle_range)  # 转换为弧度

        # 创建掩码矩阵
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        dx = x - x_center
        dy = y - y_center
        distances = np.sqrt(dx ** 2 + dy ** 2)
        angles = np.arctan2(dy, dx)

        # 判断哪些像素在圆弧区域内
        mask = (distances <= radius) & (angle_min <= angles) & (angles <= angle_max)

        # 应用掩码，将指定区域设为透明（-1）
        for z in range(depth):
            spectral_stack[:, :, z][mask] = -1

        return spectral_stack

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

    def visualize_3d_volume_pyvista(
            self,
            output_path: Path = Path("./spectral_3d_volume_pyvista.png"),
            html_path: Path = Path("./spectral_3d_volume_pyvista.html"),
            clim: tuple = None,
            resample: bool = False,
            angles: list = None,
            **kwargs,
    ):
        """
        使用 PyVista 生成3D体积渲染可视化，并保存为PNG和HTML文件。

        :param output_path: 可视化图像的保存路径（PNG文件）。
        :param html_path: 交互式可视化的HTML文件保存路径。
        :param clim: 颜色范围的元组 (min, max)。如果为 None，则自动计算。
        :param resample: ...
        """
        if not self.load_data:
            logging.warning("No image data loaded. Cannot perform 3D visualization.")
            return

        # 假设所有图像的尺寸相同
        sample_wavelength = self.wavelengths[0]
        image_shape = self.load_data[sample_wavelength].shape
        rows, cols = image_shape
        original_z = np.array(self.wavelengths)  # Original Z-axis (non-uniform)

        # 定义新的均匀 Z 轴
        num_slices = len(self.wavelengths)  # 定义目标切片数量，根据需要调整
        target_z = np.linspace(original_z.min(), original_z.max(), num_slices)

        # 创建3D数据体
        spectral_stack = np.zeros((rows, cols, len(original_z)))

        for idx, wavelength in enumerate(self.wavelengths):
            spectral_stack[:, :, idx] = self.load_data[wavelength]

        if resample:
            # 重新采样 spectral_stack 到均匀 Z 轴
            logging.info("开始重新采样 spectral_stack 到均匀 Z 轴...")
            resampled_stack = self.resample_spectral_stack(spectral_stack, original_z, target_z)
            logging.info("重新采样完成。")
        else:
            logging.info("跳过重新采样步骤。")
            resampled_stack = spectral_stack

        # 对圆柱型数据进行切片处理
        logging.info("开始对数据进行切片处理...")
        resampled_stack = self.slice_cylinder_quarter(
            resampled_stack,
            center=(cols / 2, rows / 2),  # 圆心为图像中心
            radius=None,  # 自动计算半径
            angle_range=angles  # 切掉圆弧
        )
        logging.info("切片处理完成。")

        # # 归一化数据
        max_intensity = np.max(resampled_stack)
        # if max_intensity > 0:
        #     resampled_stack = resampled_stack / max_intensity

        # 设置颜色范围
        if clim is not None:
            color_range = clim
        else:
            # 基于百分位数动态计算颜色范围
            lower_percentile = np.percentile(resampled_stack, 5)
            upper_percentile = np.percentile(resampled_stack, 95)
            color_range = (0, upper_percentile)

        # 创建 PyVista 的 ImageData
        grid = pv.ImageData()
        grid.dimensions = resampled_stack.shape
        grid.spacing = (1, 1, (target_z[1] - target_z[0]) * rows / 80)  # 假设 X 和 Y 间距为1，Z 间距为均匀
        grid.origin = (0, 0, target_z[0])

        # 将数据添加为 'intensity' 标量
        grid.point_data["intensity"] = resampled_stack.flatten(order="F")  # 使用 point_data 代替 point_arrays

        # 创建绘图器
        plotter = pv.Plotter()

        # 添加体积渲染，并设置颜色范围

        logging.info("开始添加体积渲染到绘图器...")
        plotter.add_volume(
            grid,
            scalars="intensity",
            # cmap=cmap,
            # opacity=opacity,
            # shade=True,
            clim=color_range,  # 设置颜色范围
            **kwargs,
        )
        logging.info("体积渲染添加完成。")

        # 添加标题
        plotter.add_title("3D Volume Rendering Spectral Visualization")

        # 渲染并保存为PNG截图
        logging.info(f"开始渲染并保存截图至 {output_path}...")
        try:
            plotter.show(screenshot=str(output_path))
        finally:
            pass
        logging.info(f"3D Volume Rendering Spectral Visualization 已保存至: {output_path}")

        # # 导出为交互式HTML文件
        # logging.info(f"开始导出交互式HTML至 {html_path}...")
        # plotter.export_html(str(html_path))
        # plotter.close()
        # logging.info(f"交互式HTML已保存至: {html_path}")

    def visualize_two_planes_intensity_map(
            self,
            angles: list,
            output_path: Path = Path("./two_planes_intensity_map.png"),
            plot_by_frequency: bool = False,
            **kwargs,
    ):
        """
        可视化两个指定角度上的面强度图，并拼接在一张2D颜色映射图上。

        :param angles: 包含两个需要提取的角度列表（以度为单位）。
        :param output_path: 颜色映射图保存路径（PNG文件）。
        :param plot_by_frequency: 是否按照频率顺序绘制（True），否则按照波长顺序绘制（False）。
        """
        if not self.load_data:
            logging.warning("No image data loaded. Cannot perform two planes intensity visualization.")
            return

        if len(angles) != 2:
            logging.warning("需要提供两个角度进行可视化。")
            return

        angle1, angle2 = angles
        logging.info(f"开始提取角度 {angle1}° 和 {angle2}° 的面强度数据。")

        # 假设所有图像的尺寸相同
        sample_wavelength = self.wavelengths[0]
        image_shape = self.load_data[sample_wavelength].shape
        rows, cols = image_shape
        center = (cols / 2, rows / 2)

        # 将角度转换为弧度
        angles_rad = [np.deg2rad(angle) for angle in angles]

        # 确定最大半径
        max_radius = np.min(center)

        # 设置沿线的采样点数量，确保为整数
        num_samples = int(max_radius)
        logging.info(f"设置采样点数量为: {num_samples}")

        # 生成采样点并提取面强度数据
        intensity_maps = []
        for phi_idx, (angle_deg, angle_rad) in enumerate(zip(angles, angles_rad)):
            logging.info(f"提取角度 {angle_deg}° 的面强度数据。")
            # 初始化一个二维数组：波长 x 采样点
            intensity_map = np.zeros((len(self.wavelengths), num_samples))
            for idx, wavelength in enumerate(self.wavelengths):
                image = self.load_data[wavelength]
                # 计算沿该角度的线性坐标
                x = center[0] + np.linspace(0, max_radius * np.cos(angle_rad), num_samples)
                y = center[1] + np.linspace(0, max_radius * np.sin(angle_rad), num_samples)

                # 使用插值获取光强值
                coords = np.vstack((y, x))
                intensity = map_coordinates(image, coords, order=1, mode='constant', cval=-1)

                # 将无效值（-1）替换为0，确保数组长度一致
                intensity_clean = np.where(intensity >= 0, intensity, 0)

                # 赋值到intensity_map
                intensity_map[idx, :] = intensity_clean
            if phi_idx % 2 == 1:
                # 水平翻转切面以匹配最终视觉效果
                intensity_map = intensity_map[:, ::-1]
            intensity_maps.append(intensity_map)
            intensity_maps.reverse()

        # 拼接两个二维数组（水平拼接）
        combined_intensity_map = np.hstack(intensity_maps)  # 或者使用 np.vstack 进行垂直拼接
        logging.info(f"拼接后的二维强度图形状: {combined_intensity_map.shape}")

        extent = [0, num_samples * 2, self.wavelengths[0], self.wavelengths[-1]]
        # 根据绘图模式调整波长排序
        if plot_by_frequency:
            logging.info("按照频率顺序（波长逆序）进行绘图。")
            # 如果按照频率绘图，逆转Y轴
            origin = 'upper'
        else:
            logging.info("按照波长顺序进行绘图。")
            # 按照波长顺序绘图
            origin = 'lower'

        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(12, 8))

        # 显示拼接后的二维数组
        im = ax.imshow(
            combined_intensity_map,
            aspect='auto',
            extent=extent,
            origin=origin,
            cmap='hot',
            vmin=0,
            **kwargs,
        )

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Intensity")

        # 设置标签和标题
        ax.set_xlabel("Incident angle")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title(f"Phi{angles[0]}° and Phi{angles[1]}° Slice Concatenated Intensity Map")

        # 保存图像
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
        logging.info(f"两个切面的面强度图已保存至: {output_path}")

    def apply_2d_filter(self, **kwargs):
        """
        对加载的每个二维数据应用滤波。
        """

        logging.info("开始对所有二维数据应用滤波器和平滑处理。")

        for wavelength, data in self.load_data.items():
            logging.info(f"处理波长 {wavelength} nm 的数据。")
            # 将2D NumPy数组转换为 DataFrame
            df = pd.DataFrame(data)
            processor = DataProcessor(data=df)

            # 应用滤波器和平滑
            processor.apply_image_filter(**kwargs)

            # 更新 load_data
            self.load_data[wavelength] = processor.data.values
            logging.info(f"波长 {wavelength} nm 的数据处理完成。")
        return self

    def apply_2d_upsample(self, **kwargs):
        """
        对加载的每个二维数据应用上采样。
        """

        logging.info("开始对所有二维数据应用上采样处理。")

        for wavelength, data in self.load_data.items():
            logging.info(f"处理波长 {wavelength} nm 的数据。")
            # 将2D NumPy数组转换为 DataFrame
            df = pd.DataFrame(data)
            processor = DataProcessor(data=df)

            # 应用上采样
            processor.upsample(**kwargs)

            processor.save_processed_data()

            # 更新 load_data
            self.load_data[wavelength] = processor.data.values
            logging.info(f"波长 {wavelength} nm 的数据上采样完成。")
        return self

    def crop_data(self, **kwargs):
        """
        对加载的每个二维数据进行裁剪。
        """

        logging.info("开始对所有二维数据进行裁剪处理。")

        for wavelength, data in self.load_data.items():
            logging.info(f"处理波长 {wavelength} nm 的数据。")
            # 将2D NumPy数组转换为 DataFrame
            df = pd.DataFrame(data)
            processor = DataProcessor(data=df)

            # 裁剪
            processor.crop_by_shape(**kwargs)

            cropped_data = processor.data.values.copy()

            cropped_data[np.isnan(cropped_data)] = -1
            # 更新 load_data
            self.load_data[wavelength] = cropped_data
            logging.info(f"波长 {wavelength} nm 的数据裁剪完成。")
        return self

    def process_3d_data(self, **kwargs):
        """
        对堆叠的三维数据进行处理（如3D高斯滤波）。
        """

        # 假设所有图像的尺寸相同
        sample_wavelength = self.wavelengths[0]
        image_shape = self.load_data[sample_wavelength].shape
        rows, cols = image_shape

        # 创建3D数据体
        spectral_stack = np.zeros((rows, cols, len(self.wavelengths)))

        for idx, wavelength in enumerate(self.wavelengths):
            spectral_stack[:, :, idx] = self.load_data[wavelength]

        # 使用 DataProcessor3D 进行3D滤波
        processor_3d = DataProcessor3D(data=spectral_stack)
        processor_3d.apply_3d_gaussian_filter(**kwargs)
        filtered_stack = processor_3d.data
        logging.info("3D高斯滤波完成。")

        # 更新 load_data
        for idx, wavelength in enumerate(self.wavelengths):
            self.load_data[wavelength] = filtered_stack[:, :, idx]
            logging.info(f"更新波长 {wavelength} nm 的数据。")