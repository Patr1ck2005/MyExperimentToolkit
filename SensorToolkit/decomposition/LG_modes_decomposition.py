import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.special import genlaguerre


# 定义 Laguerre-Gaussian 基函数
# 计算LG模函数
def lg_mode(l, p, r, phi, w0, lambda_=100, z=0):
    k = 2 * np.pi / lambda_  # 波数
    z_R = np.pi * w0 ** 2 / lambda_  # Rayleigh length
    R_z = np.inf if z == 0 else z * (1 + (np.pi * w0 ** 2 / lambda_) ** 2 / z ** 2) ** 0.5
    w_z = w0 * np.sqrt(1 + (z / z_R) ** 2)  # 光束半径
    p_z = np.arctan(z / (np.pi * w0 ** 2 / lambda_))  # 相位

    # 拉盖尔多项式
    Lp = genlaguerre(p, abs(l))  # 拉盖尔多项式的值
    C_l_p = np.sqrt(2 * math.factorial(p) / (np.pi * math.factorial(p + abs(l))))  # 归一化系数

    # 计算LG模
    factor = C_l_p/w_z * (np.sqrt(2) * r / w_z) ** abs(l) * np.exp(-r ** 2 / w_z ** 2) * Lp(2 * r ** 2 / w_z ** 2)
    phase = np.exp(1j * l * phi)
    exp_term = np.exp(-1j * k * z) * np.exp(1j * p_z)

    lg_beam = factor * phase * exp_term

    return lg_beam


# 定义计算展开系数的函数
def compute_coefficients(U, X, Y, u0, n_max, p_max):
    """
    计算涡旋光 U 与 Laguerre-Gaussian 基函数的展开系数
    """
    R = np.sqrt(X ** 2 + Y ** 2)
    Phi = np.arctan2(Y, X)

    coefficients = np.zeros((2*n_max+1, p_max+1), dtype=complex)

    for n in range(-n_max, n_max+1):
        for p in range(p_max+1):
            # 计算每个基函数与目标函数的内积 (即系数)
            LGn_p = lg_mode(n, p, R, Phi, u0)
            if DEBUG := False:
                plt.figure(figsize=(8, 4))
                plt.subplot(121)
                plt.imshow(np.abs(LGn_p))
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(np.abs(U))
                plt.colorbar()
                plt.show()
            coef = np.sum(U * np.conj(LGn_p)) * np.mean(np.diff(x)) * np.mean(np.diff(y))
            coefficients[n+n_max, p] = coef

    return coefficients


# 定义计算展开结果的函数
def compute_expansion(coefficients, X, Y, u0, n_max, p_max):
    """
    根据展开系数计算涡旋光的展开结果
    """
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)

    # 初始化展开结果
    U_expansion = np.zeros_like(X, dtype=complex)

    for n in range(-n_max, n_max+1):
        for p in range(p_max+1):
            # 计算每个 Laguerre-Gaussian 基函数
            LGn_p = lg_mode(n, p, r, phi, u0)

            # 累加到展开结果中
            U_expansion += coefficients[n+n_max, p] * LGn_p

    return U_expansion


# 示例使用
if __name__ == '__main__':
    DEBUG = False
    show = False
    if DEBUG:
        # 假设我们已有的复振幅数据U，这里使用一个示例来创建一个假的U
        x = np.linspace(-1, 1, 128)  # x坐标
        y = np.linspace(-1, 1, 128)  # y坐标
        X, Y = np.meshgrid(x, y)  # 创建网格
        R = np.sqrt(X ** 2 + Y ** 2)
        phi = np.arctan2(Y, X)
        # 假设是2阶涡旋光
        U = np.exp(-1j * phi * 2) * (np.exp(-(R / 45) ** 2) + 0.00*np.cos(4*phi))
        # 加入球面相位畸变
        U *= np.exp(-1j * R / 0.1)
        original_radius = 0.9
        U[np.where(R > original_radius)] = 0
    else:
        # 1511, 1519, 1528
        wavelength = 1560
        # U_intensity = np.load(rf'D:\DELL\Documents\ExperimentDataToolkit\phase_extrator\rsl\-{wavelength}nm-processed-extracted_intensity.npy')
        # U_phase = np.load(rf'D:\DELL\Documents\ExperimentDataToolkit\phase_extrator\rsl\-{wavelength}nm-processed-extracted_phase.npy')
        # U_intensity = np.load(rf'D:\DELL\Documents\myPlots\data\low_loss\NIR-mystructure-conversion-efficiency-low_loss-Si-194.67THz-1540.0nm-25deg-E0.7971-kx=-1.32-1.32-51_ky=-1.32-1.32-51.npy')
        # U_phase = np.load(rf'D:\DELL\Documents\myPlots\data\low_loss\NIR-mystructure-phase-reflected-low_loss-Si-194.67THz-1540.0nm-25deg-E0.7971-kx=-1.32-1.32-51_ky=-1.32-1.32-51.npy')

        # U_intensity = np.load(rf'D:\DELL\Documents\myPlots\data\low_loss\NIR-mystructure-conversion-efficiency-low_loss-Si-193.41THz-1550.0nm-25deg-E0.8329-kx=-1.32-1.32-51_ky=-1.32-1.32-51.npy')
        # U_phase = np.load(rf'D:\DELL\Documents\myPlots\data\low_loss\NIR-mystructure-phase-reflected-low_loss-Si-193.41THz-1550.0nm-25deg-E0.8329-kx=-1.32-1.32-51_ky=-1.32-1.32-51.npy')


        U_intensity = np.load(rf'D:\DELL\Documents\myPlots\data\low_loss\NIR-mystructure-conversion-efficiency-low_loss-Si-192.17THz-1560.0nm-25deg-E0.8124-kx=-1.32-1.32-51_ky=-1.32-1.32-51.npy')
        U_phase = np.load(rf'D:\DELL\Documents\myPlots\data\low_loss\NIR-mystructure-phase-reflected-low_loss-Si-192.17THz-1560.0nm-25deg-E0.8124-kx=-1.32-1.32-51_ky=-1.32-1.32-51.npy')
        U = np.sqrt(U_intensity)*np.exp(1j*U_phase)
        x = np.linspace(-1, 1, U.shape[0])
        y = np.linspace(-1, 1, U.shape[1])
        X, Y = np.meshgrid(y, x)
        R = np.sqrt(X**2+Y**2)
        original_radius = 0.5
        U[R > original_radius] = 0
        # U[R>0.5]=0
        # U[R<0.4]=0

    # for u0 in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:  # 选择适当的 spot-size
    for u0 in [0.4]:  # 选择适当的 spot-size
        save_name = f'LG-u0={u0}-{wavelength}'
        u0 *= original_radius  # 放缩使单位归一化
        # 计算展开系数
        n_max = 10  # 拓扑荷数
        n_lst = np.arange(-n_max, n_max+1)
        p_max = 8  # 角模数
        coefficients = compute_coefficients(U, X, Y, u0, n_max, p_max)
        coefficients_intensity = np.abs(coefficients)**2
        # 计算每一行的总和和
        purity_array = np.sum(coefficients_intensity, axis=1)/np.sum(coefficients_intensity)
        max_purity = purity_array.max()
        print(f'max {max_purity} at {purity_array.argmax()}')


        oneD_data = np.column_stack((n_lst, purity_array))
        fig, ax = plt.subplots(figsize=(6, 3))
        # 绘制柱状图
        plt.grid(True, axis='y')
        plt.bar(oneD_data[:, 0], oneD_data[:, 1]*100, width=0.8, color='black', edgecolor='black', label=f'MAX{max_purity:.3f}')
        ax.legend(loc='upper right')
        ax.set_xlim(-10, 10)
        ax.set_ylim(0, 100)
        # ax.set_xlabel('l (OAM)')
        # ax.set_ylabel('OAM purity (%)')
        ax.set_xticks([-10, 0, 10], [])
        ax.set_yticks([0, 50, 90, 99, 100], [])
        plt.savefig(f'./rsl/{save_name}_purity.png', dpi=300, bbox_inches='tight', pad_inches=0)

        # 可视化展开系数的l和p坐标图
        fig, ax = plt.subplots(figsize=(8, 6))
        np.save(f'./rsl/{save_name}-coefficients_abs.npy', coefficients_intensity)
        plt.colorbar(ax.imshow(coefficients_intensity,
                               aspect='auto',
                               origin='lower',
                               cmap='gray',
                               # norm=LogNorm(),
                               ))
        ax.set_xticks(np.arange(p_max))
        ax.set_yticks(np.arange(2*n_max+1))
        ax.set_xticklabels(range(p_max))
        ax.set_yticklabels(range(-n_max, n_max+1))
        ax.set_xlabel('n (azimuthal index)')
        ax.set_ylabel('l (OAM)')
        ax.set_title('Laguerre-Gaussian Coefficients (Intensity)')
        plt.savefig(f'./rsl/{save_name}-coefficients_abs.png', dpi=300)
        plt.show() if show else plt.close()

        # 计算展开结果
        U_expansion = compute_expansion(coefficients, X, Y, u0, n_max, p_max)

        # 可视化原始涡旋光与展开结果
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.imshow(np.abs(U) ** 2, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
        plt.title('Original Vortex Beam (Intensity)')
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(np.abs(U_expansion) ** 2, extent=[x.min(), x.max(), y.min(), y.max()], cmap='gray')
        plt.title(f'Expanded Vortex Beam (Intensity) - MAX_N={n_max}, MAX_P={p_max}')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(f'./rsl/{save_name}-rebuild.png', dpi=300)
        plt.show() if show else plt.close()
