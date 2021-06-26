import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pandas_datareader as web
from matplotlib.ticker import FuncFormatter


import tkinter as tk
from tkinter.filedialog import askopenfilename
import pandas as pd

# root = tk.Tk()
# root.withdraw()  # Prevents the Tkinter window to come up
# exlpath = askopenfilename()
# root.destroy()
# print(exlpath)
# df = pd.read_excel(exlpath)


import tkinter as tk
from tkinter.filedialog import askopenfilename
#import pandas as pd


def import_csv_data():
    global v
    csv_file_path = askopenfilename()
    print(csv_file_path)
    v.set(csv_file_path)
    #df = pd.read_csv(csv_file_path)


root = tk.Tk()
tk.Label(root, text='File Path').grid(row=0, column=0)
v = tk.StringVar()
entry = tk.Entry(root, textvariable=v).grid(row=0, column=1)
tk.Button(root, text='Browse Data Set',
          command=import_csv_data).grid(row=1, column=0)
tk.Button(root, text='Close', command=root.destroy).grid(row=1, column=1)
root.mainloop()
