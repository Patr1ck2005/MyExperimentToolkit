import numpy as np


# 计算折射角
def snell_law(theta_i, n_i, n_t):
    # theta_i: 入射角（弧度）
    # n1: 入射介质的折射率
    # n2: 折射介质的折射率
    sin_theta_t = (n_i / n_t) * np.sin(theta_i)
    theta_t = np.arcsin(sin_theta_t)
    return theta_t


# 计算反射系数（平行偏振和垂直偏振）
def fresnel_reflection_coefficients(theta_i, n_i, n_t):
    # theta_i: 入射角（弧度）
    # n1: 入射介质的折射率
    # n2: 折射介质的折射率
    theta_t = snell_law(theta_i, n_i, n_t)

    # 垂直偏振反射系数 (r_s)
    r_s = (n_i * np.cos(theta_i) - n_t * np.cos(theta_t)) / (n_i * np.cos(theta_i) + n_t * np.cos(theta_t))

    # 平行偏振反射系数 (r_p)
    r_p = (n_t * np.cos(theta_i) - n_i * np.cos(theta_t)) / (n_t * np.cos(theta_i) + n_i * np.cos(theta_t))

    return theta_t, r_s, r_p


# 计算出射偏振态琼斯矢量
def reflected_jones_vector(jones_in, r_s, r_p):
    # jones_in: 入射光的琼斯矢量
    # r_s: 垂直偏振反射系数
    # r_p: 平行偏振反射系数

    # 反射光的琼斯矢量，s偏振分量乘以 r_s，p偏振分量乘以 r_p
    jones_out = np.matmul(np.array([[r_s, 0], [0, r_p]]), jones_in)

    return jones_out


# 主函数
def fresnel_reflection(theta_i, jones_in, n_i, n_j):
    # theta_i: 入射角（弧度）
    # jones_in: 入射光的琼斯矢量（例如：[Es, Ep]）
    # n_i: 入射介质的复折射率
    # n_j: 折射介质的复折射率

    theta_t, r_s, r_p = fresnel_reflection_coefficients(theta_i, n_i, n_j)
    jones_out = reflected_jones_vector(jones_in, r_s, r_p)

    return theta_t, jones_out


if __name__ == '__main__':
    theta_i = np.deg2rad(22.5)
    n_i = 1  # 入射介质折射率
    n_j = 1.44+14.56j  # 折射介质折射率
    jones_in = np.array([1, 1j], dtype=complex)  # 入射偏振态
    # jones_in = np.array([1, 0], dtype=complex)  # 入射偏振态
    # jones_in = np.array([0, 1], dtype=complex)  # 入射偏振态
    total_in = np.sqrt(np.sum(np.abs(jones_in) ** 2))
    # 入射归一化
    jones_in /= total_in

    theta_t, jones_out = fresnel_reflection(theta_i, jones_in, n_i, n_j)

    if isinstance(theta_t, complex):
        print(f"无折射角")
    else:
        print(f"折射角: {np.degrees(theta_t):.5f}°")
    print(f"出射偏振态琼斯矢量: {jones_out}")
    total = np.sum(np.abs(jones_out) ** 2)
    print(f"出射光强: {total:.5f}")


