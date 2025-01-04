# MyExperimentToolkit 🚀

## Overview 📖

**MyExperimentToolkit** 是一个为科研实验设计的光强分布数据分析工具包，专注于光学实验数据的批量处理和高维光谱分析。其核心包含两个分析模块：
- **Optical Intensity Analyzer**：专注于光强图像的批量处理与统计分析。
- **Spectral Analyzer**：提供光谱数据的高级分析和3D可视化功能。

该工具包配备了高效的数据加载、处理、保存与可视化的底层架构，为光学实验提供一站式解决方案。

---

## Key Features ✨

- **🔄 批量处理支持**：支持多个光强分布图的批量处理，提取光谱信息。
- **✂️ 灵活裁剪**：支持对光强分布图进行圆形、矩形区域的裁剪操作。
- **🖼️ 多样化图像滤波**：提供多种滤波方式（高斯滤波、中值滤波、拉普拉斯边缘检测等）。
- **🌈 高维光谱分析**：
  - 提供基于波长的光谱数据排序。
  - 支持生成3D光谱分布图和多平面切片可视化。
- **📁 多格式支持**：支持多种图像格式（.bmp, .png, .jpg, .csv）和灵活的元数据提取。

---

## Installation 🛠️

### Requirements 📋
- Python >= 3.8
- 依赖库可通过以下方式安装：

```bash
pip install -r requirements.txt
```

### Installation Steps

1. 克隆项目：
   ```bash
   git clone https://github.com/Patr1ck2005/MyExperimentToolkit.git
   cd MyExperimentToolkit
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 运行示例代码验证安装是否成功。

---

## Modules Overview 🔍

### 1. Optical Intensity Analyzer 🎛️

**Optical Intensity Analyzer** 用于对实验的光强分布图进行批量分析，包括元数据提取、裁剪、平均强度计算等。

#### 核心功能：
- **📦 批量处理**：处理指定文件夹下的所有光强图。
- **📜 元数据提取**：自动从文件名提取波长、标签信息。
- **✂️ 裁剪与过滤**：支持基于圆形或矩形区域的裁剪。
- **📊 输出结果**：
  - 生成裁剪后的光强分布图。
  - 将结果以 CSV 格式输出，包含波长、标签和平均强度。

#### 示例代码：

```python
from core.optical_intensity_analyzer import OpticalIntensityAnalyzer

# 初始化分析器
analyzer = OpticalIntensityAnalyzer(
    input_dir="path_to_images",
    output_csv_path="output/results.csv",
    crop_shape_params={
        'center_row': 0.5,
        'center_col': 0.5,
        'radius': 0.3,
        'shape': 'circle',
        'relative': True
    }
)

# 批量处理图像
analyzer.process_all()
```

---

### 2. Spectral Analyzer 🌌

**Spectral Analyzer** 提供高级光谱数据分析工具，包括3D光谱体积渲染和多角度切片图。

#### 核心功能：
- **🌟 3D光谱渲染**：基于裁剪后的光强图像生成3D体积渲染，直观显示光谱分布。
- **📐 光谱切片分析**：支持多角度切片生成二维强度分布图。
- **📈 光谱排序与归一化**：支持按波长升序/降序排序，并对光谱强度进行归一化处理。

#### 示例代码：

```python
from core.spectral_analysis import SpectralAnalyzer

# 初始化光谱分析器
analyzer = SpectralAnalyzer(
    images_dir="path_to_cropped_images",
    boundary_na=0.42,
    wavelength_order='descending'
)

# 生成3D光谱体积渲染
analyzer.run_3d_volume_visualization_pyvista(
    output_path="output/spectral_volume.png",
    html_path="output/spectral_volume.html"
)

# 生成二维切片强度图
analyzer.run_two_planes_intensity_map_visualization(
    angles=[0, 90],
    output_path="output/two_planes_intensity_map.png"
)
```

## Example Workflow ⚙️

```python
from core.optical_intensity_analyzer import OpticalIntensityAnalyzer
from core.spectral_analysis import SpectralAnalyzer

# Step 1: 批量处理光强图像
intensity_analyzer = OpticalIntensityAnalyzer(
    input_dir="path_to_raw_images",
    output_csv_path="output/processed_data.csv",
    crop_shape_params={
        'center_row': 0.5,
        'center_col': 0.5,
        'radius': 0.3,
        'shape': 'circle',
        'relative': True
    }
)
intensity_analyzer.process_all()

# Step 2: 加载裁剪后的数据进行光谱分析
spectral_analyzer = SpectralAnalyzer(
    images_dir="path_to_cropped_images",
    boundary_na=0.42
)
spectral_analyzer.run_3d_volume_visualization_pyvista(
    output_path="output/spectral_3d_volume.png"
)
```

---

## Contributing 🤝

如果您有兴趣贡献代码或发现问题，请通过 GitHub 提交 Issue 或 Pull Request。

---

## License 📜

**本项目不提供任何开源协议**。如有合作需求，请联系维护者。

---
