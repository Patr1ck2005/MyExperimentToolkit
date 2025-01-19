import numpy as np
from numpy.fft import fft2, fftshift, ifftshift, ifft2


# 计算傅里叶变换并加入相移
def calculate_fourier(image, shift_x=0, shift_y=0):
    """
    计算傅里叶变换，并根据给定的平移量在频域中加入相移。
    shift_x 和 shift_y 为图像中心的平移量
    """
    # 计算傅里叶变换并中心化
    F = np.fft.fftshift(np.fft.fft2(image))  # 中心化傅里叶变换

    ny, nx = image.shape
    u = np.fft.fftfreq(nx) * nx  # 水平方向频率坐标
    v = np.fft.fftfreq(ny) * ny  # 垂直方向频率坐标

    U, V = np.meshgrid(u, v)  # 生成频率网格

    # 修正频率坐标，使得相移计算正确
    phase_shift = np.exp(1j * 2 * np.pi * (U * shift_x / nx + V * shift_y / ny))  # 加入平移因子

    F_shifted = F * phase_shift  # 应用平移因子到傅里叶变换

    return F_shifted



def apply_filter(F, loc_x, loc_y, radius, ny, nx):
    """
    根据给定的滤波参数对傅里叶谱进行滤波。

    参数:
        F: 傅里叶谱
        loc_x: 滤波中心 x 坐标
        loc_y: 滤波中心 y 坐标
        radius: 滤波半径
        ny, nx: 图像尺寸

    返回:
        F_filtered: 滤波后的傅里叶谱
        interference_filtered: 逆变换后的空间域图像
    """
    # 创建掩码
    y, x = np.ogrid[:ny, :nx]
    mask = np.zeros((ny, nx))
    mask_area = (y - loc_y) ** 2 + (x - loc_x) ** 2 <= radius ** 2
    mask[mask_area] = 1

    # 应用掩码
    F_filtered = F * mask
    F_filtered = np.roll(F_filtered, shift=(ny // 2 - loc_y, nx // 2 - loc_x), axis=(0, 1))

    # 逆傅里叶变换
    interference_filtered = ifft2(ifftshift(F_filtered))
    return F_filtered, interference_filtered
