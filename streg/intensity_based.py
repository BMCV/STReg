import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import math
from skimage.filters import gaussian
from tqdm import tqdm

from streg.utils import Normalize


def calculate_center_of_mass(image):
    x = np.sum(image, axis=0)
    idx_x = np.arange(len(x))
    coord_x = np.sum(idx_x * x) / np.sum(x)

    y = np.sum(image, axis=1)
    idx_y = np.arange(len(y))
    coord_y = np.sum(idx_y * y) / np.sum(y)

    return coord_x, coord_y


def get_params(alpha, translation, scale, shear):
    angle = (2 * np.pi) / 360 * alpha
    angle = torch.tensor(float(angle), requires_grad=True)

    t1 = torch.tensor(float(translation[0]), requires_grad=True)
    t2 = torch.tensor(float(translation[1]), requires_grad=True)

    sh1 = torch.tensor(float(shear[0]), requires_grad=True)
    sh2 = torch.tensor(float(shear[1]), requires_grad=True)

    sc1 = torch.tensor(float(scale[0]), requires_grad=True)
    sc2 = torch.tensor(float(scale[1]), requires_grad=True)

    params = [angle, t1, t2, sh1, sh2, sc1, sc2]

    return params


def get_affine_matrix(params):

    id = torch.eye(3)
    rot = torch.eye(3)

    s = torch.sin(math.pi * F.tanh(params[0]))
    c = torch.cos(math.pi * F.tanh(params[0]))

    rot[0, 0] = c
    rot[0, 1] = -s
    rot[1, 0] = s
    rot[1, 1] = c

    trans = torch.eye(3)
    trans[0, -1] = 2 * F.tanh(params[1])
    trans[1, -1] = 2 * F.tanh(params[2])

    sh = torch.eye(3)
    sh[0, 0] = F.tanh(params[3]) * F.tanh(params[4]) + 1
    sh[0, 1] = F.tanh(params[3])
    sh[1, 0] = F.tanh(params[4])

    sc = torch.eye(3)
    sc[0, 0] = 2 ** F.tanh(params[6])
    sc[1, 1] = 2 ** F.tanh(params[6])

    matrix = id @ rot @ trans @ sh @ sc
    return matrix[:2][None]


class Intensity_Registration():
    def __init__(self, resize=(512, 512), init_alpha=0, init_trans=[0, 0], init_scale=[0, 0], init_shear=[0, 0], lr=1e-3, optimizer='Adam'):

        self.resize = resize
        self.params = get_params(alpha=init_alpha, translation=init_trans, scale=init_scale, shear=init_shear)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.losses = []
        self.lr = []

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.params, lr=lr, betas=(0.5, 0.9))
        else:
            self.optimizer = optim.SGD(self.params, lr=lr, momentum=0.5)

        self.registration_matrix = None

    def get_matrix(self,):
        tmp = np.eye(3)
        R = get_affine_matrix(self.params)
        tmp[0] = R[0, 0].detach().cpu().numpy()
        tmp[1] = R[0, 1].detach().cpu().numpy()
        return tmp

    def prepare_img(self, img, resize=None, normalize=False, smooth_sigma=None):

        img_ = img.copy().astype(np.float32)

        if smooth_sigma:
            img_ = gaussian(img_, sigma=smooth_sigma)

        img_ = torch.from_numpy(img_)[None, None].float()
        if resize is not None:
            img_ = F.interpolate(img_, size=resize, mode='bilinear')

        if normalize:
            img_ = Normalize(img_)

        return img_.to(self.device)

    def optimize(self, N, staining, genes, crit=nn.MSELoss(), smooth_stain=5, smooth_gene=5):
        print('Start intensity-based registration...')
        moving = self.prepare_img(staining, self.resize, normalize=True, smooth_sigma=smooth_stain)
        fixed = self.prepare_img(genes, self.resize, normalize=True, smooth_sigma=smooth_gene)
        weight = 5 * self.prepare_img(genes, self.resize, normalize=True) + 1

        pbar = tqdm(range(N))

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=N, eta_min=1e-6)

        for i in pbar:
            R = get_affine_matrix(self.params).to(self.device)
            grid = F.affine_grid(R, moving.size(), align_corners=True)
            x_rot = F.grid_sample(moving, grid, align_corners=True, mode='bilinear', padding_mode='zeros')

            loss = crit(x_rot, fixed, weight)

            pbar.set_description(f'Loss - {loss.item():.4f}')

            self.losses.append(loss.item())

            self.lr.append(self.optimizer.param_groups[0]['lr'])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        self.registration_matrix = get_affine_matrix(self.params)

    def shift_center_of_mass(self, staining, genes):
        x_stain, y_stain = calculate_center_of_mass(staining)
        x_gene, y_gene = calculate_center_of_mass(genes)

        return x_gene - x_stain, y_gene - y_stain

    def register(self, img):
        img_ = torch.from_numpy(img.astype(np.float32)).type(torch.float32)
        img_ = img_[None, None]
        R = get_affine_matrix(self.params)
        grid = F.affine_grid(R.type(torch.float32), img_.size(), align_corners=True)
        out = F.grid_sample(img_, grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        return out[0, 0].detach().numpy()
