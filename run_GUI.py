import os
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk

from streg.utils import load_images, ToolTip
from streg.GUI.frames import State, ImageFrame, AdjustImage, ManualRegistration, IntensityBased, TracklineBased, Close, Save


class ImageEditor(tk.Toplevel):
    def __init__(self, parent, staining_image, gene_image, max_size=None):
        super().__init__(parent)
        font_style = ("Arial", 10, "bold")

        self.title('Spatial Transcriptomics Registration')

        self.state = State(self, staining_image, gene_image, max_size)

        self.image_frame = ImageFrame(self,
                                      tk.Label(self, text='Gray: Staining Image (Moving)\n\nRed: Gene Image (Fixed)', justify='left'),
                                      self.state)
        self.image_frame.grid(row=0, column=1, rowspan=5)

        self.adjust_image = AdjustImage(self,
                                        tk.Label(self, text='Adjust Brightness/Contrast', font=font_style),
                                        self.state
                                        )
        self.adjust_image.grid(row=0, column=2, sticky='new', rowspan=5)

        self.manual_registration = ManualRegistration(self,
                                                      tk.Label(self, text='Step 1: Manual Registration (optional)', font=font_style),
                                                      self.state)
        self.manual_registration.grid(row=0, column=0, sticky='ew', pady=10)

        self.intensity_registration = IntensityBased(self,
                                                     tk.Label(self, text='Step 2: Intensity-Based Registration', font=font_style),
                                                     self.state)
        self.intensity_registration.grid(row=1, column=0, sticky='ew', pady=10)

        self.trackline_registration = TracklineBased(self,
                                                     tk.Label(self, text='Step 3: Trackline-Based Registration', font=font_style),
                                                     self.state)
        self.trackline_registration.grid(row=2, column=0, sticky='ew', pady=10)

        self.save = Save(self,
                         tk.Label(self, text='Step 4: Save Results', font=font_style),
                         self.state)
        self.save.grid(row=3, column=0, sticky='ew', pady=10)

        self.close = Close(self,
                           tk.Label(self, text='', font=font_style),
                           self.state)
        self.close.grid(row=4, column=0, sticky='ew', pady=10)

        for i in range(self.grid_size()[0]):
            self.columnconfigure(i, weight=1)
        for j in range(self.grid_size()[1]):
            self.rowconfigure(j, weight=1)

        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

    def update_image(self):
        self.state.tk_image = ImageTk.PhotoImage(self.image_frame.create_composite(self.state))
        self.state.canvas.itemconfig(self.image_frame.image_id, image=self.state.tk_image)


class MainWindow(tk.Tk):
    def __init__(self,):
        super().__init__()
        self.title('Spatial Transcriptomics Registration')

        padx = 0
        pady = 5

        label_gene = tk.Label(self, text='Gene Image')
        label_gene.grid(row=0, column=0, sticky='ew', padx=padx, pady=pady)

        entry_gene = tk.Entry(self, width=50)
        entry_gene.grid(row=0, column=1, sticky='ew', padx=padx, pady=pady)

        bttn_gene = tk.Button(self, text='Browse', command=lambda: self.browse(entry_gene, 'tabular'))
        bttn_gene.grid(row=0, column=2, sticky='ew', padx=padx, pady=pady)

        label_stain = tk.Label(self, text='Staining Image')
        label_stain.grid(row=1, column=0, sticky='ew', padx=padx, pady=pady)

        entry_stain = tk.Entry(self, width=50)
        entry_stain.grid(row=1, column=1, sticky='ew', padx=padx, pady=pady)

        bttn_stain = tk.Button(self, text='Browse', command=lambda: self.browse(entry_stain, 'image'))
        bttn_stain.grid(row=1, column=2, sticky='ew', padx=padx, pady=pady)

        frame1 = tk.LabelFrame(self, text='Settings')
        frame1.grid(row=2, column=0, columnspan=2, padx=padx, pady=pady)

        label_size = tk.Label(frame1, text='Resize images:')
        label_size.grid(row=0, column=0, sticky='ew', padx=padx, pady=pady)

        entry_size = tk.Entry(frame1)
        entry_size.grid(row=0, column=1, sticky='ew', padx=padx, pady=pady)
        entry_size.insert(0, 512)

        bttn_open = tk.Button(self, text='Open', command=lambda: self.open_files(entry_gene, entry_stain, entry_size.get()))
        bttn_open.grid(row=0, column=3, sticky='ew', padx=padx, pady=pady)

        bttn_test_img = tk.Button(self, text='Load example image', command=lambda: self.load_example(entry_gene, entry_stain))
        bttn_test_img.grid(row=2, column=2, columnspan=2, sticky='ew', padx=padx, pady=pady)

        quit_button = tk.Button(self, text='Quit', command=lambda: self.quit())
        quit_button.grid(row=1, column=3, sticky='ew', padx=padx, pady=pady)

        self.update_idletasks()
        self.minsize(self.winfo_width(), self.winfo_height())

    def browse(self, entry, filetype):
        if filetype == 'tabular':
            ft = [['csv', '*.csv'], ['csv', '*.csv.gz'], ['tsv', '*.tsv'], ['tsv', '*.tsv.gz']]
        elif filetype == 'image':
            ft = [['tif', '*.tiff'], ['tif', '*.tif']]

        path = filedialog.askopenfilename(filetypes=ft, initialdir='~/')
        if path:
            entry.delete(0, tk.END)
            entry.insert(0, path)

    def open_files(self, entry_gene, entry_stain, max_size=512):
        if isinstance(max_size, str):
            max_size = int(max_size)

        max_size = max(max_size, 256)
        gene_path = entry_gene.get()
        stain_path = entry_stain.get()

        assert len(gene_path) != 0, 'No gene data selected...'
        assert len(stain_path) != 0, 'No staining image selected...'
        assert gene_path.endswith(('.csv','.tsv', '.csv.gz', '.tsv.gz')), 'Wrong filetype for gene image, ".csv" or ".tsv" required...'
        assert stain_path.endswith('.tif'), 'Wrong filetype for staining image. ".tif" required...'

        staining, genes = load_images(gene_path, stain_path)

        ImageEditor(self, staining, genes, max_size=max_size)

    def load_example(self, entry_gene, entry_stain):
        gene_test = os.path.abspath('test_data/gene_matrix_test.csv.gz')
        stain_test = os.path.abspath('test_data/staining_image_test.tif')
        entry_gene.insert(0, gene_test)
        entry_stain.insert(0, stain_test)


if __name__ == '__main__':
    app = MainWindow()
    app.mainloop()
