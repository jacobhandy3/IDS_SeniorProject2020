import tkinter as tk
import main
import NNcode

class Application(tk.Frame):
    def init(self, master=None):
        super().init(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.runFirstDataset = tk.Button(self)
        self.runFirstDataset.config( height = 5, width =40 )
        self.runFirstDataset["text"] = "First Dataset"
        self.runFirstDataset["command"] = self.firstDataset
        self.runFirstDataset.pack(side="top")

        self.runSecondDataset = tk.Button(self)
        self.runSecondDataset.config( height = 5, width =40 )
        self.runSecondDataset["text"] = "Second Dataset"
        self.runSecondDataset["command"] = self.secondDataset
        self.runSecondDataset.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def firstDataset(self):
      NNcode.NNanalysis(path=r"DataSets\CIC-IDS-2017", header=0, indexCol=None, 
                  mapped=main.attacksCIC, rowL=main.CICrows, Xmax=77, labelCol=78, attackNum=15)

    def secondDataset(self):
      NNcode.NNanalysis(path=r"DataSets\UNSW-NB15", header=None, indexCol=None,
                    mapped=main.mappingUNSW,rowL=main.UNSWcols, Xmax=44, labelCol=45, attackNum=10,
                    dropFeats=[1,3], missReplacement=["Benign"],missCols=[47])



root = tk.Tk()
app = Application(master=root)
app.mainloop()