# core/filename_parser.py

import logging
from typing import Dict, List, Optional


class FilenameParser:
    """
    提供从文件名中提取波长和标签信息的工具类。
    """

    def __init__(self, delimiter: str = '-'):
        """
        初始化文件名解析器。

        :param delimiter: 分隔符，用于分割文件名中的不同部分。
        """
        self.delimiter = delimiter
        logging.info(f"初始化 FilenameParser，分隔符: '{self.delimiter}'")

    def extract_info(self, filename: str) -> Dict[str, Optional[object]]:
        """
        从文件名中提取波长和其他标签信息。

        :param filename: 图像文件名。
        :return: 包含波长和标签的字典。
        """
        # 去除扩展名
        name_without_ext = filename.rsplit('.', 1)[0]
        parts = name_without_ext.split(self.delimiter)
        info = {
            'wavelength_nm': None,
            'filename': filename
        }
        labels_from_filename: List[str] = []

        # 如果无法解析为波长，将所有部分作为标签
        labels_from_filename.extend(parts)
        logging.info(f"文件名 {filename} 中提取标签: {labels_from_filename}")
        for label in labels_from_filename:
            try:
                wavelength = float(parts[0])
                info['wavelength_nm'] = wavelength
                logging.info(f"解析第一个部分提取到波长: {wavelength} nm 从文件名: {filename}")
            except ValueError:
                pass  # 如果不是波长，继续尝试其他部分

            if label.endswith('nm'):
                try:
                    wavelength = float(label[:-2])
                    info['wavelength_nm'] = wavelength
                    logging.info(f"提取到波长: {wavelength} nm 从文件名: {filename}")
                    labels_from_filename.remove(label)
                    break
                except ValueError:
                    pass  # 如果不是波长，继续尝试其他部分
        info['labels_from_filename'] = labels_from_filename
        return info
