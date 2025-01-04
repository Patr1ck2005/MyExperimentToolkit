# core/optical_intensity_analyzer.py

from pathlib import Path
import logging
from SensorToolkit.core.data_loading import ImageDataLoader
from SensorToolkit.core.data_processing import DataProcessor
from SensorToolkit.core.data_saving import CSVSaver
from SensorToolkit.utils.filename_parser import FilenameParser
import pandas as pd


class OpticalIntensityAnalyzer:
    """
    专注于处理科研实验测量的光强分布图案的处理类。
    """

    def __init__(
        self,
        input_dir: Path,
        output_csv_path: Path,
        crop_shape_params: dict,
        labels: dict = None,
        filename_delimiter: str = '-'
    ):
        """
        初始化分析器。

        :param input_dir: 输入图像文件所在的目录。
        :param output_csv_path: 输出结果 CSV 文件的路径。
        :param crop_shape_params: 裁剪参数字典。
        :param labels: 可选的标签字典，键为文件名，值为标签。
        :param filename_delimiter: 文件名中用于分割不同部分的分隔符。
        """
        self.input_dir = input_dir
        self.output_csv_path = output_csv_path
        self.crop_shape_params = crop_shape_params
        self.labels = labels or {}
        self.results = []
        self.filename_parser = FilenameParser(delimiter=filename_delimiter)  # 使用 FilenameParser
        logging.info(f"初始化 OpticalIntensityAnalyzer，输入目录: {self.input_dir}, 输出 CSV: {self.output_csv_path}")

    def extract_info(self, filename: str) -> dict:
        """
        从文件名中提取波长和其他标签信息。

        :param filename: 图像文件名。
        :return: 包含波长和标签的字典。
        """
        return self.filename_parser.extract_info(filename)  # 使用 FilenameParser

    def add_label(self, filename: str, label: str):
        """
        手动添加标签。

        :param filename: 图像文件名。
        :param label: 标签内容。
        """
        self.labels[filename] = label
        logging.info(f"为文件 {filename} 添加标签: {label}")

    def process_image(self, image_file: Path):
        """
        处理单个图像文件。

        :param image_file: 图像文件路径。
        """
        filename = image_file.name
        info = self.extract_info(filename)
        wavelength = info['wavelength_nm']
        labels_from_filename = info['labels_from_filename']
        manual_label = self.labels.get(filename, "")

        # 合并标签
        combined_labels = labels_from_filename.copy()
        if manual_label:
            combined_labels.append(manual_label)

        # 加载图像数据
        loader = ImageDataLoader(image_file)
        data = loader.load_data()

        # 定义保存处理后图像的路径
        temp_dir = Path("./temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        # 处理数据
        processor = DataProcessor(data)
        (processor
         # .apply_image_filter(filter_type='gaussian', sigma=10)
         # .apply_image_filter(filter_type='laplace')
         # .apply_image_filter(filter_type='lowpass', cutoff=20)
         # .apply_image_filter(filter_type='highpass', cutoff=20)
         # .apply_image_filter(filter_type='median', size=50)
         .crop_by_shape(
            center_row=self.crop_shape_params['center_row'],
            center_col=self.crop_shape_params['center_col'],
            radius=self.crop_shape_params['radius'],
            inner_radius=self.crop_shape_params.get('inner_radius', 0),
            shape=self.crop_shape_params['shape'],
            relative=self.crop_shape_params['relative'],
            )
         )

        file_name_without_ext = image_file.stem
        file_extension = image_file.suffix
        image_save_path = temp_dir / f"{file_name_without_ext}-processed.png"  # 更新保存路径为PNG
        processor.save_processed_image(save_path=image_save_path)
        processor.reset_coordinates()
        avg_intensity = processor.calculate_average_intensity()

        # 保存结果
        result = {
            'filename': filename,
            'wavelength_nm': wavelength,
            'labels': ','.join(combined_labels) if combined_labels else "",
            'average_intensity': avg_intensity
        }
        self.results.append(result)
        logging.info(f"文件 {filename} 处理完成，平均光强: {avg_intensity}, 裁剪图像保存至: {image_save_path}")

    def process_all(self):
        """
        批量处理输入目录中的所有图像文件。
        """
        image_files = list(self.input_dir.glob("*.bmp"))  # 处理BMP文件
        if not image_files:
            logging.warning(f"在 {self.input_dir} 中未找到任何 BMP 文件。")
            return

        logging.info(f"开始批量处理 {len(image_files)} 个文件。")
        for image_file in image_files:
            try:
                self.process_image(image_file)
            except Exception as e:
                logging.error(f"处理文件 {image_file.name} 时发生错误: {e}")
                continue

        # 保存所有结果到 CSV
        if self.results:
            df = pd.DataFrame(self.results)
            saver = CSVSaver()
            saver.save(df, self.output_csv_path)
            logging.info(f"所有处理结果已保存至 {self.output_csv_path}")
        else:
            logging.warning("没有任何结果需要保存。")
