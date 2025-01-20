from pathlib import Path
import logging
from SensorToolkit.core.data_loading import ImageDataLoader
from SensorToolkit.core.data_processing import DataProcessor
from SensorToolkit.core.data_saving import CSVSaver
from SensorToolkit.utils.filename_parser import FilenameParser
import pandas as pd
from typing import Optional, List, Dict
from contextlib import contextmanager

# 定义文件扩展名常量
IMAGE_EXTENSION = "*.png"


class OpticalIntensityAnalyzer:
    """
    专注于处理科研实验测量的光强分布图案的处理类。
    """

    def __init__(
            self,
            input_dir: Optional[Path] = None,
            input_dirs: Optional[List[Path]] = None,
            output_csv_path: Optional[Path] = None,
            crop_shape_params: Optional[Dict] = None,
            labels: Optional[Dict[str, str]] = None,
            filename_delimiter: str = '-'
    ):
        self.processor = None
        self.input_dirs = input_dirs if input_dirs else [input_dir]
        self.output_csv_path = output_csv_path
        self.crop_shape_params = crop_shape_params
        self.labels = labels or {}
        self.results = []
        self.filename_parser = FilenameParser(delimiter=filename_delimiter)
        logging.info(f"初始化 OpticalIntensityAnalyzer，输入目录: {self.input_dirs}, 输出 CSV: {self.output_csv_path}")

    def extract_info(self, filename: str) -> dict:
        """ 从文件名中提取波长和其他标签信息 """
        return self.filename_parser.extract_info(filename)

    def add_label(self, filename: str, label: str):
        """ 手动添加标签，避免重复添加 """
        if filename not in self.labels:
            self.labels[filename] = label
            logging.info(f"为文件 {filename} 添加标签: {label}")
        else:
            logging.warning(f"文件 {filename} 已经有标签，跳过添加。")

    @contextmanager
    def temp_dir(self):
        """ 临时目录上下文管理器 """
        temp_dir = Path("./temp")
        temp_dir.mkdir(parents=True, exist_ok=True)  # 创建临时目录
        yield temp_dir  # 返回临时目录，不做任何删除操作

    def process_image(self, image_file: Path):
        """ 处理单个图像文件 """
        filename = image_file.name
        info = self.extract_info(filename)
        wavelength = info['wavelength_nm']
        labels_from_filename = info['labels_from_filename']
        manual_label = self.labels.get(filename, "")
        combined_labels = labels_from_filename + ([manual_label] if manual_label else [])

        loader = ImageDataLoader(image_file)
        data = loader.load_data()
        self.processor = DataProcessor(data)

        with self.temp_dir() as temp_dir:
            file_name_without_ext = image_file.stem
            image_save_path = temp_dir / f"{file_name_without_ext}-processed.png"

            self.processor.crop_by_shape(
                center_row=self.crop_shape_params['center_row'],
                center_col=self.crop_shape_params['center_col'],
                radius=self.crop_shape_params['radius'],
                inner_radius=self.crop_shape_params.get('inner_radius', 0),
                shape=self.crop_shape_params['shape'],
                relative=self.crop_shape_params['relative']
            )
            self.processor.save_processed_data(save_path=image_save_path)
            self.processor.reset_coordinates()
            avg_intensity = self.processor.calculate_average_intensity()
            logging.info(f"文件 {filename} 处理完成，平均光强: {avg_intensity}, 裁剪图像保存至: {image_save_path}")

            result = {
                'dir': self.current_input_dir.name,
                'filename': filename,
                'wavelength_nm': wavelength,
                'labels': ','.join(combined_labels),
                'average_intensity': avg_intensity
            }
            self.results.append(result)

    def statistic_image(self, image_file: Path, statistics_params: Dict = None):
        """ 统计单个图像文件的光强信息 """
        filename = image_file.name
        info = self.extract_info(filename)
        wavelength = info['wavelength_nm']
        labels_from_filename = info['labels_from_filename']
        manual_label = self.labels.get(filename, "")
        combined_labels = labels_from_filename + ([manual_label] if manual_label else [])

        loader = ImageDataLoader(image_file)
        data = loader.load_data()
        self.processor = DataProcessor(data)

        avg_intensity = self.processor.calculate_average_intensity() / 255
        avg_intensities_in_ranges = {}

        if statistics_params:
            NA_list = statistics_params.get('NA', [])
            for NA in NA_list:
                radius = self.processor.NA2radius(NA, full_na=0.42, full_radius=0.5)
                avg_intensities_in_ranges[f'avg_intensity_NA{NA}'] = self.processor.calculate_avg_intensity_in_radius_range(0, radius)/255

        logging.info(f"文件 {filename} 统计完成，平均光强: {avg_intensity}, 其他范围光强平均值: {avg_intensities_in_ranges}")

        result = {
            'dir': self.current_input_dir.name,
            'filename': filename,
            'wavelength_nm': wavelength,
            'labels': ','.join(combined_labels),
            'average_intensity': avg_intensity
        }

        result.update(avg_intensities_in_ranges)
        self.results.append(result)

    def process_images_in_dir(self, image_files: List[Path], process: bool, statistics_params: Dict = None):
        """ 批量处理或统计指定目录中的所有图像文件 """
        for image_file in image_files:
            try:
                if process:
                    self.process_image(image_file)
                else:
                    self.statistic_image(image_file, statistics_params)
            except Exception as e:
                logging.error(f"处理文件 {image_file.name} 时发生错误: {e}")

    def process_all(self, statistics_params: Dict = None):
        """ 批量处理所有输入目录中的所有图像文件 """
        for input_dir in self.input_dirs:
            self.current_input_dir = input_dir
            logging.info(f"开始处理目录: {input_dir}")
            image_files = list(self.current_input_dir.glob(IMAGE_EXTENSION))
            if not image_files:
                logging.warning(f"在 {self.current_input_dir} 中未找到任何 PNG 文件。")
                return

            logging.info(f"开始批量处理 {len(image_files)} 个文件。")
            self.process_images_in_dir(image_files, process=True, statistics_params=statistics_params)

        if self.results:
            df = pd.DataFrame(self.results)
            saver = CSVSaver()
            saver.save(df, self.output_csv_path)
            logging.info(f"所有处理结果已保存至 {self.output_csv_path}")
        else:
            logging.warning("没有任何结果需要保存。")

    def statistic_all(self, statistics_params: Dict = None):
        """ 批量统计输入目录中的所有图像文件 """
        for input_dir in self.input_dirs:
            self.current_input_dir = input_dir
            image_files = list(self.current_input_dir.glob(IMAGE_EXTENSION))
            if not image_files:
                logging.warning(f"在 {self.current_input_dir} 中未找到任何 PNG 文件。")
                return

            logging.info(f"开始批量统计 {len(image_files)} 个文件。")
            self.process_images_in_dir(image_files, process=False, statistics_params=statistics_params)

        if self.results:
            df = pd.DataFrame(self.results)
            saver = CSVSaver()
            saver.save(df, self.output_csv_path)
            logging.info(f"所有统计结果已保存至 {self.output_csv_path}")
        else:
            logging.warning("没有任何结果需要保存。")
