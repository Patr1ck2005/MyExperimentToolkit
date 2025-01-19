from tkinter import Tk

from SensorToolkit.phase_extrator.phase_extractor_window import FourierApp

root = Tk()
app = FourierApp(root, image_path=r'./data/1508~1528-cropped/-1521nm-processed.png')
root.mainloop()
