import kmm, k3m
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
from PIL import Image, ImageTk
from tkinter import filedialog
from PIL import Image
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import os

class App(Frame):

    def __init__(self, parent):
        super(App, self).__init__()

        self.master.state('normal')
        self.master.title("APBI")
        self.master.rowconfigure(4, weight=1)
        self.master.columnconfigure(4, weight=1)
        self.grid(sticky=W + E + N + S)

        self.parent = parent

        self.button = Button(self, text="Choose image", command=self.load_file, width=30)
        self.button.grid(row=0, column=0, sticky=W)

        self.buttonGenerate = Button(self, text="KMM", command=self.kmm, width=30)
        self.buttonGenerate.grid(row=0, column=1, sticky=W)

        self.buttonGenerate = Button(self, text="K3M", command=self.k3m, width=30)
        self.buttonGenerate.grid(row=0, column=2, sticky=W)

    def kmm(self):
        out = kmm.kmm(self.input_image.copy())
        imshow(out, cmap=plt.cm.binary)

        # Very dummy way because problems with converting np 1 bit color array to Pillow.Image
        plt.savefig('temp.png')
        out = Image.open('temp.png')
        os.remove("temp.png")

        img = ImageTk.PhotoImage(out)
        self.generated = Label(self, image=img)
        self.generated.image = img
        self.generated.grid(row=2, column=0, sticky=E)

    def k3m(self):
        out = k3m.k3m(self.input_image.copy())

        img = ImageTk.PhotoImage(out)
        self.generated = Label(self, image=img)
        self.generated.image = img
        self.generated.grid(row=2, column=0, sticky=E)

    def load_file(self):
        fname = filedialog.askopenfilename(initialdir='.')
        img = Image.open(fname)
        img.thumbnail((self.winfo_screenwidth() * .4, self.winfo_screenheight() * .4))
        self.input_image = img
        img = ImageTk.PhotoImage(img)
        self.photo = Label(self.parent, image=img)
        self.photo.image = img
        self.photo.grid(row=1, column=0, sticky=W)


root = Tk()
app = App(root)
root.mainloop()
