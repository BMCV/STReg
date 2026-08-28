import os
import numpy as np
import tkinter as tk
from tifffile import imread
import pandas as pd


class SliceBuilder:
    def __init__(self, img, patch_shape, stride_shape):
        self._img_slices = self._build_slices(img, patch_shape, stride_shape)

    @property
    def img_slices(self):
        return self._img_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        slices = []
        if dataset.ndim == 3:
            in_channels, i_y, i_x = dataset.shape
        else:
            i_y, i_x = dataset.shape

        k_y, k_x = patch_shape
        s_y, s_x = stride_shape

        y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
        for y in y_steps:
            x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
            for x in x_steps:
                slice_idx = (
                    slice(y, y + k_y),
                    slice(x, x + k_x)
                )
                if dataset.ndim == 3:
                    slice_idx = (slice(0, in_channels),) + slice_idx
                slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


def shift_coords(df):
    df['x'] = df['x'] - df['x'].min()
    df['y'] = df['y'] - df['y'].min()
    return df


def Normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def Standardize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def toUint8(x):
    return (255 * Normalize(x)).astype(np.uint8)


def cleanse_matrix(df):
    return df.loc[df['UMI_Count'] > 0]


def genes2image(df, height, width):
    gene_img = np.zeros((height, width), dtype=np.uint8)
    coords_x = df['x']
    coords_y = df['y']
    umi_count = df['UMI_Count']

    np.add.at(gene_img, (coords_y, coords_x), umi_count)
    return gene_img


def load_images(gene_path, staining_path):
    if staining_path.endswith('.tif'):
        staining_image = imread(staining_path)

    if 'tsv' in gene_path.split('.'):
        df = pd.read_csv(gene_path, sep='\t', comment='#')
    else:
        df = pd.read_csv(gene_path)

    mapping = {name: 'UMI_Count' for name in ['UMI_Count', 'MIDCounts', 'MIDCount']}
    df.rename(columns=mapping, inplace=True)

    df = shift_coords(df)
    df = cleanse_matrix(df)

    h_s, w_s = staining_image.shape
    h_g, w_g = df['y'].max() + 1, df['x'].max() + 1

    if h_g > h_s:
        staining_image = np.pad(staining_image, ((0, h_g - h_s), (0, 0)))
    if w_g > w_s:
        staining_image = np.pad(staining_image, ((0, 0), (0, w_g - w_s)))

    x_max = np.max((df['x'].max() + 1, staining_image.shape[1]))
    y_max = np.max((df['y'].max() + 1, staining_image.shape[0]))

    gene_image = genes2image(df, y_max, x_max)

    return staining_image, gene_image


def crop(genes, staining):
    indices = np.argwhere(genes != 0)
    min_x = indices[:, 0].min()
    max_x = indices[:, 0].max()
    min_y = indices[:, 1].min()
    max_y = indices[:, 1].max()

    staining_cropped = staining[min_x:max_x, min_y:max_y]
    gene_cropped = genes[min_x:max_x, min_y:max_y]

    return gene_cropped, staining_cropped, (min_x, max_x, min_y, max_y)


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.waittime = 500
        self.tw = None

        self.widget.bind('<Enter>', self.show)
        self.widget.bind('<Leave>', self.hide)

    def show(self, event=None):
        if self.tw is not None:
            return

        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25

        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f'+{x}+{y}')
        label = tk.Label(
            self.tw, text=self.text, background='#ffffff', relief='solid', borderwidth=1
        )

        label.pack(ipadx=3, ipady=1)

    def hide(self, event=None):
        if self.tw is not None:
            self.tw.destroy()
            self.tw = None


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
