import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, Frame, Label, Button, DoubleVar, Scale, HORIZONTAL
from PIL import Image

from SensorToolkit.phase_extrator.phase_extract_cal import calculate_fourier, apply_filter


# 主应用程序
class FourierApp:
    def __init__(self, root, image_path, initial_value: dict = None):
        self.root = root
        self.root.title("Fourier Transform Visualization")
        self.filestem = image_path.split('/')[-1].split('.')[0]
        # 加载图像并计算初始傅里叶变换
        img = Image.open(image_path)
        img = img.convert('L')  # 转换为灰度图像
        self.image = np.array(img)
        self.F = calculate_fourier(self.image)

        self.ny, self.nx = self.image.shape

        # 初始化参数
        self.loc_x = DoubleVar(value=self.nx // 2)
        self.loc_y = DoubleVar(value=self.ny // 2)
        self.radius = DoubleVar(value=10)
        self.shift_x = DoubleVar(value=0.0)  # 图像平移的 x 方向
        self.shift_y = DoubleVar(value=0.0)  # 图像平移的 y 方向
        if initial_value:
            self.loc_x.set(initial_value.get('loc_x', self.nx // 2))
            self.loc_y.set(initial_value.get('loc_y', self.ny // 2))
            self.radius.set(initial_value.get('radius', 10))
            self.shift_x.set(initial_value.get('shift_x', 0.0))
            self.shift_y.set(initial_value.get('shift_y', 0.0))

        # 创建控制面板
        control_frame = Frame(root)
        control_frame.grid(row=0, column=0, padx=10, pady=10)

        # loc_x 控制
        Label(control_frame, text="loc_x").grid(row=0, column=0)
        self.loc_x_scale = Scale(control_frame, from_=0, to=self.nx, orient=HORIZONTAL, variable=self.loc_x, command=self.update_canvas)
        self.loc_x_scale.grid(row=0, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.loc_x, 1)).grid(row=0, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.loc_x, -1)).grid(row=0, column=5)

        # loc_y 控制
        Label(control_frame, text="loc_y").grid(row=1, column=0)
        self.loc_y_scale = Scale(control_frame, from_=0, to=self.ny, orient=HORIZONTAL, variable=self.loc_y, command=self.update_canvas)
        self.loc_y_scale.grid(row=1, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.loc_y, 1)).grid(row=1, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.loc_y, -1)).grid(row=1, column=5)

        # radius 控制
        Label(control_frame, text="radius").grid(row=2, column=0)
        self.radius_scale = Scale(control_frame, from_=1, to=50, orient=HORIZONTAL, variable=self.radius, command=self.update_canvas)
        self.radius_scale.grid(row=2, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.radius, 1)).grid(row=2, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.radius, -1)).grid(row=2, column=5)

        # 平移 控制
        Label(control_frame, text="Shift X").grid(row=3, column=0)
        self.shift_x_scale = Scale(control_frame, from_=-50, to=50, orient=HORIZONTAL, variable=self.shift_x, command=self.update_canvas)
        self.shift_x_scale.grid(row=3, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.shift_x, 1)).grid(row=3, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.shift_x, -1)).grid(row=3, column=5)

        Label(control_frame, text="Shift Y").grid(row=4, column=0)
        self.shift_y_scale = Scale(control_frame, from_=-50, to=50, orient=HORIZONTAL, variable=self.shift_y, command=self.update_canvas)
        self.shift_y_scale.grid(row=4, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.shift_y, 1)).grid(row=4, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.shift_y, -1)).grid(row=4, column=5)

        # 初始化 Matplotlib 图形
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1)

        # 添加放大工具
        toolbar_frame = Frame(root)
        toolbar_frame.grid(row=1, column=1, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # 添加保存按钮
        save_button = Button(root, text="Save Images", command=self.save_images)
        save_button.grid(row=2, column=0, padx=10, pady=10)

        # 初始化显示
        self.update_canvas()

    def save_images(self):
        """
        保存四个子图为独立的图片，去掉标题、坐标轴和其他文字，仅保存图像。
        """
        # 设置保存的文件名
        filenames = [
            f'./rsl/{self.filestem}-original_fourier_spectrum.png',
            f'./rsl/{self.filestem}-filtered_fourier_spectrum.png',
            f'./rsl/{self.filestem}-extracted_intensity.png',
            f'./rsl/{self.filestem}-extracted_phase.png'
        ]

        # 保存每个子图
        for i, ax in enumerate(self.axes.flat):
            # 创建新的 Figure 对象
            fig = plt.figure(figsize=(6, 6))  # 可以根据需要调整大小
            # 将子图的内容复制到新的 Figure
            new_ax = fig.add_subplot(111)

            # 获取子图的内容
            # 只处理图像艺术对象
            for artist in ax.get_children():
                # 如果 artist 是图像对象，复制其内容
                if isinstance(artist, mpimg.AxesImage):
                    new_ax.imshow(artist.get_array(), cmap=artist.get_cmap())

            # 去除坐标轴和标题
            new_ax.axis('off')

            # 保存子图
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(filenames[i], dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close(fig)  # 关闭当前的 Figure，以释放内存

    def adjust_param(self, param, step):
        """
        微调参数值，并同步更新滑动条。
        """
        param.set(param.get() + step)
        self.update_canvas()

    def update_canvas(self, *args):
        """
        更新画布内容，包括傅里叶谱和提取后的图像。
        """
        loc_x = int(self.loc_x.get())
        loc_y = int(self.loc_y.get())
        radius = int(self.radius.get())
        shift_x = self.shift_x.get()  # 获取 x 方向的平移量
        shift_y = self.shift_y.get()  # 获取 y 方向的平移量

        # 计算滤波结果
        F_shifted = calculate_fourier(self.image, shift_x, shift_y)
        F_filtered, interference_filtered = apply_filter(F_shifted, loc_x, loc_y, radius, self.ny, self.nx)

        # 更新傅里叶谱
        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(np.log(np.abs(F_shifted) + 1), cmap='gray')
        self.axes[0, 0].set_title('Original Fourier Spectrum with Shift')
        self.axes[0, 0].axis('off')

        self.axes[0, 1].clear()
        self.axes[0, 1].imshow(np.log(np.abs(F_filtered) + 1), cmap='gray')
        self.axes[0, 1].set_title('Filtered Fourier Spectrum')
        self.axes[0, 1].axis('off')

        # 更新提取的图像
        self.axes[1, 0].clear()
        self.axes[1, 0].imshow(np.abs(interference_filtered) ** 2, cmap='gray')
        self.axes[1, 0].set_title('Extracted Intensity')
        self.axes[1, 0].axis('off')

        self.axes[1, 1].clear()
        self.axes[1, 1].imshow(np.angle(interference_filtered), cmap='twilight')
        self.axes[1, 1].set_title('Extracted Phase')
        self.axes[1, 1].axis('off')

        self.canvas.draw()



if __name__ == "__main__":
    root = Tk()
    # app = FourierApp(root, image_path='./artificial_pattern.png')
    # app = FourierApp(root, image_path=r'./-1518nm-processed.png')
    app = FourierApp(root, image_path=r'./-1518nm.png')
    # app = FourierApp(root, image_path=r'1.png')
    # app = FourierApp(root, image_path='./interference_cropped.bmp')
    # app = FourierApp(root, image_path='./interference_1.bmp')
    # app = FourierApp(root, image_path='./interference_2.bmp')
    # app = FourierApp(root, image_path='./interference_3.bmp')
    root.mainloop()
