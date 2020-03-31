import tkinter as tk
from tkinter import filedialog
#import main
#import NNcode

folder_path = ""
""" header = None
indexCol=None
colL = None
Xmax = None
labelCol = None
attackNum = None
dropFeats = None
missReplacement = None
missCols = None

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.browse_button()

    def browse_button(self):
        global folder_path
        filename = filedialog.askdirectory()
        folder_path.set(filename)
        print(filename)



root = tk.Tk()
dirname = filedialog.askdirectory(parent=root, initialdir="/",
                                        title='Please select a directory')
if(len(dirname)>0):
    print("You chose %s"%dirname)
app = Application(master=root)
app.mainloop() """

#                   #
#   NEXT EXAMPLE    #
#                   #

fields = 'Header', 'Index Column', 'Text Columns(csv)','Label Column', 'No. of Attacks', 'Columns to Exclude(csv)', 'Missing Data Replacement(csv)', 'Missing Data Columns(csv)'

def fetch(entries):
    for entry in entries:
        field = entry[0]
        text  = entry[1].get()
        print('%s: "%s"' % (field, text)) 

def browse_button():
    global folder_path
    filename = filedialog.askdirectory()
    folder_path.set(filename)
    print(filename)

def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

if __name__ == '__main__':
    root = tk.Tk()
    ents = makeform(root, fields)
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))   
    b1 = tk.Button(root, text='Browse',
                  command=lambda : browse_button())
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text='Quit', command=root.quit)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()