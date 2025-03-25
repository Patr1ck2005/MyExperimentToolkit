from tkinter import Tk

from SensorToolkit.phase_extrator.phase_extractor_window import FourierApp

root = Tk()
initial_values = {
    'loc_x': 380,
    'loc_y': 352,
    'radius': 12
}
# 1511, 1519, 1528
app = FourierApp(root, image_path=r'./data/1508~1528-cropped_bigger/-1519nm-processed.png', initial_value=initial_values)
root.mainloop()
