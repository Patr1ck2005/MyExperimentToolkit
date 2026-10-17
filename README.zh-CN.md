# MyExperimentToolkit

**历史光强图像与光谱分析工具集。** 本仓库保留公开 Python 版本，包含光强处理、光谱可视化和序列处理工具。

[English](README.md) · [原有说明与示例](docs/README-history.md)

## 快照索引

| 路径 | 此公开版本中的用途 |
|---|---|
| [SensorToolkit/](SensorToolkit/) | 共用加载、处理与可视化组件 |
| `intensity_analysis.py`、`intensity_analysis-sim.py` | 光强分析入口 |
| `spectral_analysis.py`、`spectral_analysis-sim.py` | 光谱分析入口 |
| `sequence_divider.py`、`sequence_process.py` | 序列处理工具 |
| [phase-extrator-app/](phase-extrator-app/) | 独立相位提取应用材料 |
| [requirements.txt](requirements.txt) | 原有依赖列表 |

## 状态与使用

这里是历史**公开**快照，不是较新私有重建版的发布。本次文档更新没有引入私有模块、实验图像、科研结果或部署数据。

原脚本覆盖图像加载、裁剪／滤波、统计及多维作图。使用前应按自己的输入核对文件名、波长元数据、数组布局及本机路径假设。没有本次新验证的一键安装器、发行包或随仓测试数据。先阅读 [旧 README](docs/README-history.md) 与相应脚本；`requirements.txt` 记录原依赖，不代表当前全部版本兼容。

本次没有启动应用，没有读取或处理实验数据。

## 权利声明

**本项目不提供任何开源协议。** 保留原有明确声明。公开可见不授予按 MIT 或其他协议使用、再分发工具集的权限；合作或复用请联系维护者。依赖继续适用各自许可。
