import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import laguerre

from PIL import Image


# 定义计算展开系数的函数
def compute_coefficients(U, X, Y, n_max):
    """
    计算涡旋光 U 与 Laguerre-Gaussian 基函数的展开系数
    """
    R = np.sqrt(X ** 2 + Y ** 2)
    Phi = np.arctan2(Y, X)

    coefficients = np.zeros((2*n_max+1, 1), dtype=complex)

    for n in range(-n_max, n_max+1):
        # 计算每个基函数与目标函数的内积 (即系数)
        basis = np.exp(-1j * n * Phi)
        coef = np.sum(U * np.conj(basis))*((X[0, 1] - X[0, 0]) * (Y[1, 0] - Y[0, 0]))/len(X)
        coefficients[n + n_max] = coef

    coefficients /= np.sum(np.abs(coefficients))
    return coefficients


# 示例使用
if __name__ == '__main__':
    DEBUG = False
    if DEBUG:
        # 假设我们已有的复振幅数据U，这里使用一个示例来创建一个假的U
        x = np.linspace(-50, 50, 512)  # x坐标
        y = np.linspace(-50, 50, 512)  # y坐标
        X, Y = np.meshgrid(x, y)  # 创建网格
        R = np.sqrt(X ** 2 + Y ** 2)
        phi = np.arctan2(Y, X)
        # 假设是2阶涡旋光
        U = np.exp(-1j * phi * 2) * (np.exp(-(R / 15) ** 2) + 0.00*np.cos(4*phi))*10  # 假设一个理想的2阶涡旋光复振幅分布
        # 加入球面相位畸变
        U *= np.exp(-1j * R / 10)
    else:
        U_intensity = np.load(r'D:\DELL\Documents\ExperimentDataToolkit\phase_extrator\rsl\-1519nm-processed-extracted_intensity.npy')
        U_phase = np.load(r'D:\DELL\Documents\ExperimentDataToolkit\phase_extrator\rsl\-1519nm-processed-extracted_phase.npy')
        U = np.sqrt(U_intensity)*np.exp(1j*U_phase)
        x = np.linspace(-1, 1, U.shape[0])
        y = np.linspace(-1, 1, U.shape[1])
        X, Y = np.meshgrid(y, x)

    plt.imshow(U.real)
    plt.show()

    # 计算展开系数
    n_max = 5
    coefficients = compute_coefficients(U, X, Y, n_max)

    abs_coefficients = np.abs(coefficients)
    # abs_coefficients /= np.sum(abs_coefficients)

    # 可视化展开系数的l和p坐标图
    fig, ax = plt.subplots(figsize=(8, 6))
    # data = np.abs(coefficients)
    ax.plot(abs_coefficients)
    ax.set_xticks(np.arange(2*n_max+1))
    ax.set_xticklabels(range(-n_max, n_max+1))
    ax.set_xlabel('l (topology number)')
    ax.set_title('Decomposition Coefficients (Intensity)')
    plt.show()

    # 可视化原始涡旋光
    plt.figure(figsize=(6, 6))

    norm = np.max(np.abs(U))
    alpha_data = np.abs(U) / norm  # 计算透明度
    phase = np.angle(U)  # 获取相位
    # 创建一个 RGB 图像（我们可以将相位值映射到颜色范围）
    phase_color = plt.cm.twilight((phase+np.pi) / (2 * np.pi))  # 将相位值映射到 twiligh 色图
    # 将 alpha_data 扩展为 RGBA 通道中的 alpha 通道
    phase_color[..., 3] = alpha_data  # 透明度信息加入到 alpha 通道
    # 显示图像
    plt.imshow(phase_color, extent=[x.min(), x.max(), y.min(), y.max()])
    plt.show()
