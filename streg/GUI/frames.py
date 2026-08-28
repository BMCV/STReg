import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox
from cv2 import warpAffine
from PIL import Image, ImageTk, ImageOps
from skimage.filters import gaussian
from scipy.ndimage import zoom
from tifffile import imwrite

from streg.utils import toUint8, ToolTip, crop
from streg.losses import MSE, NCC, NMI
from streg.intensity_based import Intensity_Registration
from streg.line_based import Line_Registration


class State:
    def __init__(self, parent, staining_image, gene_image, max_size):
        self.padx = 0
        self.pady = 5

        self.brightness_stain = 0
        self.contrast_stain = 1
        self.brightness_gene = 0
        self.contrast_gene = 1

        self.gene_image = gene_image.copy()
        self.staining_image = staining_image.copy()

        self.max_size = max_size

        self.manually_registered = tk.BooleanVar(value=False)

        self.total_matrix = np.eye(3)


class ImageFrame(tk.LabelFrame):
    def __init__(self, parent, label, state):
        super().__init__(parent, labelwidget=label, relief='flat')

        if state.max_size is not None:
            h, w = state.staining_image.shape
            if np.max((h, w)) > state.max_size:
                factor = state.max_size / np.max((h, w))
                staining_image = zoom(gaussian(state.staining_image.copy(), sigma=3, preserve_range=True), factor, order=1)
                gene_image = zoom(gaussian(state.gene_image.copy(), sigma=3, preserve_range=True), factor, order=1)

                state.scaling_matrix = np.eye(3)
                state.scaling_matrix[0, 0] = factor
                state.scaling_matrix[1, 1] = factor
            else:
                staining_image = state.staining_image.copy()
                gene_image = state.gene_image.copy()
                state.scaling_matrix = np.eye(3)

        state.gene_image_ds = toUint8(gene_image.copy())
        state.fixed_image = Image.fromarray(toUint8(gene_image.copy()))

        state.staining_image_ds = toUint8(staining_image.copy())
        state.moving_image = Image.fromarray(toUint8(staining_image.copy()))

        state.composite_image = self.create_composite(state)
        state.tk_image = ImageTk.PhotoImage(state.composite_image)

        state.canvas = tk.Canvas(self, width=state.moving_image.width, height=state.moving_image.height)
        state.canvas.grid(row=0, column=0)

        self.image_id = state.canvas.create_image(0, 0, anchor=tk.NW, image=state.tk_image)

    def create_composite(self, state):
        fixed = Image.fromarray(np.clip(np.array(state.fixed_image) * state.contrast_gene + state.brightness_gene, 0, 255).astype(np.uint8))
        moving = Image.fromarray(np.clip(np.array(state.moving_image) * state.contrast_stain + state.brightness_stain, 0, 255).astype(np.uint8))

        fixed_rgb = ImageOps.colorize(fixed, black='black', white='red')
        moving_rgb = ImageOps.colorize(moving, black='black', white='white')

        moving_rgba = moving_rgb.convert('RGBA')
        moving_rgba.putalpha(128)

        fixed_rgba = fixed_rgb.convert('RGBA')

        composite = Image.alpha_composite(fixed_rgba, moving_rgba)
        return composite.convert('RGB')


class AdjustImage(tk.LabelFrame):
    def __init__(self, parent, label, state):
        super().__init__(parent, labelwidget=label)
        self.parent = parent
        self.state = state
        label_contrast_stain = tk.Label(self, text='Contrast staining image:')
        label_contrast_stain.grid(row=0, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.slider_contrast_stain = tk.Scale(self, from_=0, to=4, resolution=0.1, orient='horizontal', command=self.adjust_contrast_stain)
        self.slider_contrast_stain.set(1)
        self.slider_contrast_stain.grid(row=0, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        label_bright_stain = tk.Label(self, text='Brightness staining image:')
        label_bright_stain.grid(row=1, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.slider_bright_stain = tk.Scale(self, from_=0, to=255, resolution=1, orient='horizontal', command=self.adjust_brightness_stain)
        self.slider_bright_stain.set(0)
        self.slider_bright_stain.grid(row=1, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        label_contrast_gene = tk.Label(self, text='Contrast gene image:')
        label_contrast_gene.grid(row=2, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.slider_contrast_gene = tk.Scale(self, from_=0, to=4, resolution=0.1, orient='horizontal', command=self.adjust_contrast_gene)
        self.slider_contrast_gene.set(1)
        self.slider_contrast_gene.grid(row=2, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        label_bright_gene = tk.Label(self, text='Brightness gene image:')
        label_bright_gene.grid(row=3, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.slider_bright_gene = tk.Scale(self, from_=0, to=255, resolution=1, orient='horizontal', command=self.adjust_brightness_gene)
        self.slider_bright_gene.set(0)
        self.slider_bright_gene.grid(row=3, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.reset_button_b_c = tk.Button(self, text='Reset', command=self.reset_b_c)
        self.reset_button_b_c.grid(row=4, column=0, columnspan=2, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        for i in range(self.grid_size()[0]):
            self.grid_columnconfigure(i, weight=1)

    def adjust_contrast_stain(self, value):
        self.state.contrast_stain = float(value)
        self.parent.update_image()

    def adjust_brightness_stain(self, value):
        self.state.brightness_stain = float(value)
        self.parent.update_image()

    def adjust_contrast_gene(self, value):
        self.state.contrast_gene = float(value)
        self.parent.update_image()

    def adjust_brightness_gene(self, value):
        self.state.brightness_gene = float(value)
        self.parent.update_image()

    def reset_b_c(self):
        self.state.brightness_stain = 0
        self.state.contrast_stain = 1
        self.state.brightness_gene = 0
        self.state.contrast_gene = 1

        self.slider_contrast_stain.set(1)
        self.slider_bright_stain.set(0)
        self.slider_contrast_gene.set(1)
        self.slider_bright_gene.set(0)

        self.parent.update_image()


class ManualRegistration(tk.LabelFrame):
    def __init__(self, parent, label, state):
        super().__init__(parent, labelwidget=label)
        self.parent = parent
        self.state = state

        self.translation_label = tk.Label(self, text='Translation: Click & Drag')
        self.translation_label.grid(row=0, column=0, columnspan=2, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.scaling_label = tk.Label(self, text='Scaling: Mousewheel')
        self.scaling_label.grid(row=1, column=0, columnspan=2, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.flip_h_button = tk.Button(self, text='Flip Horizontal', command=self.flip_horizontal)
        self.flip_h_button.grid(row=2, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.flip_v_button = tk.Button(self, text='Flip Vertical', command=self.flip_vertical)
        self.flip_v_button.grid(row=2, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.rotate_button = tk.Button(self, text='Rotate 90', command=self.rotate_image)
        self.rotate_button.grid(row=3, column=0, columnspan=2, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        for i in range(self.grid_size()[0]):
            self.grid_columnconfigure(i, weight=1)

        self.state.canvas.bind('<ButtonPress-1>', self.start_drag)
        self.state.canvas.bind('<B1-Motion>', self.drag)

        self.state.canvas.bind('<MouseWheel>', self.zoom)
        self.state.canvas.bind('<Button-4>', self.zoom)
        self.state.canvas.bind('<Button-5>', self.zoom)

    def flip_horizontal(self):
        trans = np.eye(3)
        trans[0, 2] = -np.array(self.state.moving_image).shape[1] / 2
        trans[1, 2] = -np.array(self.state.moving_image).shape[0] / 2

        tmp = np.eye(3)
        tmp[0, 0] = -1

        flip = np.linalg.inv(trans) @ tmp @ trans
        self.state.total_matrix = flip @ self.state.total_matrix
        self.transform()

    def flip_vertical(self):
        trans = np.eye(3)
        trans[0, 2] = -np.array(self.state.moving_image).shape[1] / 2
        trans[1, 2] = -np.array(self.state.moving_image).shape[0] / 2

        tmp = np.eye(3)
        tmp[1, 1] = -1

        flip = np.linalg.inv(trans) @ tmp @ trans
        self.state.total_matrix = flip @ self.state.total_matrix
        self.transform()

    def start_drag(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def drag(self, event):
        dy = event.x - self.start_x
        dx = event.y - self.start_y

        tmp = np.eye(3)
        tmp[0, 2] = dy
        tmp[1, 2] = dx

        self.state.total_matrix = tmp @ self.state.total_matrix

        self.transform()

        self.start_x = event.x
        self.start_y = event.y

        self.state.manually_registered.set(True)

    def zoom(self, event):

        trans = np.eye(3)
        trans[0, 2] = -np.array(self.state.moving_image).shape[1] / 2
        trans[1, 2] = -np.array(self.state.moving_image).shape[0] / 2

        tmp = np.eye(3)

        if event.num == 4:
            tmp[0, 0] = 0.9
            tmp[1, 1] = 0.9
        elif event.num == 5:
            tmp[0, 0] = 1.1
            tmp[1, 1] = 1.1

        elif event.delta > 0:
            tmp[0, 0] = 0.9
            tmp[1, 1] = 0.9
        elif event.delta < 0:
            tmp[0, 0] = 1.1
            tmp[1, 1] = 1.1

        scale = np.linalg.inv(trans) @ tmp @ trans

        self.state.total_matrix = scale @ self.state.total_matrix

        self.transform()

        self.state.manually_registered.set(True)

    def rotate_image(self):
        alpha = 90
        rad = 2 * np.pi / 360 * alpha

        trans = np.eye(3)
        trans[0, 2] = -np.array(self.state.moving_image).shape[1] / 2
        trans[1, 2] = -np.array(self.state.moving_image).shape[0] / 2

        tmp = np.eye(3)
        tmp[0, 0] = np.cos(rad)
        tmp[0, 1] = -np.sin(rad)
        tmp[1, 0] = np.sin(rad)
        tmp[1, 1] = np.cos(rad)

        rot = np.linalg.inv(trans) @ tmp @ trans

        self.state.total_matrix = rot @ self.state.total_matrix
        self.transform()

    def transform(self,):
        out = warpAffine(np.array(self.state.staining_image_ds), self.state.total_matrix[:2], (np.array(self.state.staining_image_ds).shape[1], np.array(self.state.staining_image_ds).shape[0]))
        self.state.moving_image = Image.fromarray(out)
        self.parent.update_image()


class IntensityBased(tk.LabelFrame):
    def __init__(self, parent, label, state):
        super().__init__(parent, labelwidget=label)
        self.parent = parent
        self.state = state

        label_com = tk.Label(self, text='Center-of-mass alignment:')
        label_com.grid(row=0, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.com_box = tk.Checkbutton(self, variable=self.state.manually_registered, onvalue=False, offvalue=True)
        self.com_box.grid(row=0, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        optimizers = ['Adam', 'SGD']
        self.opt_variable = tk.StringVar(self)
        self.opt_variable.set(optimizers[0])
        label_opt = tk.Label(self, text='Optimizer:')
        label_opt.grid(row=1, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.dropdown_optim = tk.OptionMenu(self, self.opt_variable, *optimizers)
        self.dropdown_optim.grid(row=1, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        losses = ['MSE', 'NCC', 'NMI']
        self.loss_variable = tk.StringVar(self)
        self.loss_variable.set(losses[0])
        label_loss = tk.Label(self, text='Loss:')
        label_loss.grid(row=2, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.dropdown_loss = tk.OptionMenu(self, self.loss_variable, *losses)
        self.dropdown_loss.grid(row=2, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        label_lr = tk.Label(self, text='Learning Rate:')
        label_lr.grid(row=3, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.entry1 = tk.Entry(self)
        self.entry1.insert(0, 0.01)
        self.entry1.grid(row=3, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        # ToolTip(label_lr, text='If there is significant distortion or the moving image disappears completely, try reducing the learning rate.')

        label_niter = tk.Label(self, text='Iterations:')
        label_niter.grid(row=4, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)
        self.entry2 = tk.Entry(self)
        self.entry2.insert(0, 1000)
        self.entry2.grid(row=4, column=1, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.int_reg_button = tk.Button(self, text='Run', command=self.intensity_registration)
        self.int_reg_button.grid(row=5, column=0, columnspan=2, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        for i in range(self.grid_size()[0]):
            self.grid_columnconfigure(i, weight=1)

    def intensity_registration(self):
        loss = self.loss_variable.get()
        lr = float(self.entry1.get())
        n_iter = int(self.entry2.get())
        optimizer = self.opt_variable.get()

        if loss == 'MSE':
            crit = MSE()
        elif loss == 'NCC':
            crit = NCC()
        elif loss == 'NMI':
            crit = NMI()

        int_reg = Intensity_Registration(resize=None, init_alpha=0, init_trans=[0, 0], lr=lr, optimizer=optimizer)

        if not self.state.manually_registered.get():
            print('Adjusting center of masses...')
            shift_x, shift_y = int_reg.shift_center_of_mass(np.array(self.state.moving_image), np.array(self.state.gene_image_ds))
            matrix_shift = np.eye(3)
            matrix_shift[0, 2] = shift_x
            matrix_shift[1, 2] = shift_y

            self.state.total_matrix = matrix_shift @ self.state.total_matrix

            out = warpAffine(np.array(self.state.staining_image_ds), self.state.total_matrix[:2], (np.array(self.state.staining_image_ds).shape[1], np.array(self.state.staining_image_ds).shape[0]))
            self.state.moving_image = Image.fromarray(out)

            self.state.manually_registered.set(True)

        int_reg.optimize(n_iter, np.array(self.state.moving_image), np.array(self.state.gene_image_ds), crit=crit, smooth_stain=None, smooth_gene=None)

        matrix = int_reg.get_matrix()
        m_shift = np.eye(3)
        m_shift[0, 2] = -np.array(self.state.moving_image).shape[1] / 2
        m_shift[1, 2] = -np.array(self.state.moving_image).shape[0] / 2
        m_scale = np.eye(3)
        m_scale[0, 0] = 2 / np.array(self.state.moving_image).shape[1]
        m_scale[1, 1] = 2 / np.array(self.state.moving_image).shape[0]

        M = m_scale @ m_shift

        m_total = np.linalg.inv(M) @ matrix @ M
        m_total = np.linalg.inv(m_total)

        self.state.total_matrix = m_total @ self.state.total_matrix

        self.transform()

        print('Intensity Registration finished...')

    def transform(self,):
        out = warpAffine(np.array(self.state.staining_image_ds), self.state.total_matrix[:2], (np.array(self.state.staining_image_ds).shape[1], np.array(self.state.staining_image_ds).shape[0]))
        self.state.moving_image = Image.fromarray(out)
        self.parent.update_image()


class TracklineBased(tk.LabelFrame):
    def __init__(self, parent, label, state):
        super().__init__(parent, labelwidget=label)
        self.parent = parent
        self.state = state

        self.line_reg_button = tk.Button(self, text='Run', command=self.line_registration)
        self.line_reg_button.grid(row=0, column=0, columnspan=2, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        for i in range(self.grid_size()[0]):
            self.grid_columnconfigure(i, weight=1)

    def line_registration(self):
        line_reg = Line_Registration(
            patch_size=(2000, 2000),
            stride=(2000, 2000),
            detection_threshold=0.05,
            order=200,
            matching_threshold=150,
            RANSAC_iterations=10000,
            RANSAC_threshold=3,
            gaussian_sigma=2,
        )

        R = np.linalg.inv(self.state.scaling_matrix) @ self.state.total_matrix @ self.state.scaling_matrix
        fullsize_staining_image_transformed = warpAffine(self.state.staining_image, R[:2], (self.state.staining_image.shape[1], self.state.staining_image.shape[0]))
        line_reg.optmize(fullsize_staining_image_transformed, self.state.gene_image)

        p = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 1]], dtype=float)

        matrix_line = p @ line_reg.registration_matrix @ p
        R_tmp = self.state.scaling_matrix @ matrix_line @  np.linalg.inv(self.state.scaling_matrix)
        self.state.total_matrix = R_tmp @ self.state.total_matrix
        self.transform()
        print('Line Registration finished...')

    def transform(self,):
        out = warpAffine(np.array(self.state.staining_image_ds), self.state.total_matrix[:2], (np.array(self.state.staining_image_ds).shape[1], np.array(self.state.staining_image_ds).shape[0]))
        self.state.moving_image = Image.fromarray(out)
        self.parent.update_image()


class Save(tk.LabelFrame):
    def __init__(self, parent, label, state):
        super().__init__(parent, labelwidget=label)
        self.parent = parent
        self.state = state

        self.save_button = tk.Button(self, text='Save...', command=self.save_img)
        self.save_button.grid(row=0, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        for i in range(self.grid_size()[0]):
            self.grid_columnconfigure(i, weight=1)

    def save_img(self,):
        out_dir = filedialog.askdirectory()

        if not out_dir:
            return

        R = np.linalg.inv(self.state.scaling_matrix) @ self.state.total_matrix @ self.state.scaling_matrix
        fullsize_staining_image_transformed = warpAffine(self.state.staining_image, R[:2], (self.state.staining_image.shape[1], self.state.staining_image.shape[0]))

        gene_out_cropped, stain_out_crooped, crop_params = crop(self.state.gene_image, fullsize_staining_image_transformed)
        transformation_parameters = {'affine_matrix': R, 'size': (self.state.staining_image.shape[1], self.state.staining_image.shape[0]), 'crop_params': crop_params}

        try:
            imwrite(out_dir + '/staining_image_registered.tif', stain_out_crooped)
            imwrite(out_dir + '/gene_image.tif', gene_out_cropped)
            np.save(out_dir + '/parameters.npy', transformation_parameters)
            messagebox.showinfo('Info', f'Saved image to: \n{out_dir}')
        except:
            messagebox.showerror('Error', f'Failed saving to:\n{out_dir}')


class Close(tk.LabelFrame):
    def __init__(self, parent, label, state):
        super().__init__(parent, labelwidget=label, relief='flat')
        self.parent = parent
        self.state = state

        self.reset_button = tk.Button(self, text='Reset all', command=self.reset)
        self.reset_button.grid(row=0, column=0, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        self.exit_button = tk.Button(self, text='Exit', command=self.exit)
        self.exit_button.grid(row=0, column=1, columnspan=2, sticky='ew', padx=self.state.padx, pady=self.state.pady)

        for i in range(self.grid_size()[0]):
            self.grid_columnconfigure(i, weight=1)

    def reset(self):
        self.state.total_matrix = np.eye(3)
        self.state.moving_image = self.state.staining_image_ds.copy()
        self.parent.update_image()
        self.state.manually_registered.set(False)

    def exit(self,):
        self.parent.quit()
