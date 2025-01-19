from tkinter import Tk

from SensorToolkit.phase_extrator.phase_extractor_window import FourierApp

root = Tk()
initial_values = {
    'loc_x': 358,
    'loc_y': 331,
    'radius': 12
}
app = FourierApp(root, image_path=r'./data/1508~1528-cropped/-1528nm-processed.png', initial_value=initial_values)
root.mainloop()
