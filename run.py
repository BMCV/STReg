import os
import numpy as np
import argparse

from tifffile import imwrite
from scipy.ndimage import zoom
from cv2 import warpAffine
from skimage.filters import gaussian

from streg.intensity_based import Intensity_Registration
from streg.line_based import Line_Registration
from streg.losses import MSE, NCC, NMI
from streg.utils import load_images, Normalize, crop, ensure_dir


def intensity_based(args, source, target, total_matrix):
    loss = args.loss
    lr = float(args.lr)
    n_iter = int(args.niter)
    optimizer = args.optimizer

    if loss == 'MSE':
        crit = MSE()
    elif loss == 'NCC':
        crit = NCC()
    elif loss == 'NMI':
        crit = NMI()

    int_reg = Intensity_Registration(resize=None, init_alpha=0, init_trans=[0, 0], lr=lr, optimizer=optimizer)

    print('Adjusting center of masses...')
    shift_x, shift_y = int_reg.shift_center_of_mass(np.array(warpAffine(np.array(source), total_matrix[:2], (np.array(source).shape[1], np.array(source).shape[0]))), np.array(target))
    matrix_shift = np.eye(3)
    matrix_shift[0, 2] = shift_x
    matrix_shift[1, 2] = shift_y

    total_matrix = matrix_shift @ total_matrix

    out = warpAffine(np.array(source), total_matrix[:2], (np.array(source).shape[1], np.array(source).shape[0]))

    int_reg.optimize(n_iter, np.array(out), np.array(target), crit=crit, smooth_stain=None, smooth_gene=None)

    matrix = int_reg.get_matrix()
    m_shift = np.eye(3)
    m_shift[0, 2] = -np.array(out).shape[1] / 2
    m_shift[1, 2] = -np.array(out).shape[0] / 2
    m_scale = np.eye(3)
    m_scale[0, 0] = 2 / np.array(out).shape[1]
    m_scale[1, 1] = 2 / np.array(out).shape[0]

    M = m_scale @ m_shift

    m_total = np.linalg.inv(M) @ matrix @ M
    m_total = np.linalg.inv(m_total)

    total_matrix = m_total @ total_matrix

    return total_matrix


def line_based(source, target, total_matrix, scaling_matrix):
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

    R = np.linalg.inv(scaling_matrix) @ total_matrix @ scaling_matrix
    fullsize_staining_image_transformed = warpAffine(source, R[:2], (source.shape[1], source.shape[0]))
    line_reg.optmize(fullsize_staining_image_transformed, target)

    p = np.array([[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=float)

    matrix_line = p @ line_reg.registration_matrix @ p
    R_tmp = scaling_matrix @ matrix_line @  np.linalg.inv(scaling_matrix)
    total_matrix = R_tmp @ total_matrix
    return total_matrix


def flip_horizontal(image):
    trans = np.eye(3)
    trans[0, 2] = -np.array(image).shape[1] / 2
    trans[1, 2] = -np.array(image).shape[0] / 2

    tmp = np.eye(3)
    tmp[0, 0] = -1

    flip = np.linalg.inv(trans) @ tmp @ trans

    return flip


def flip_vertical(image):
    trans = np.eye(3)
    trans[0, 2] = -np.array(image).shape[1] / 2
    trans[1, 2] = -np.array(image).shape[0] / 2

    tmp = np.eye(3)
    tmp[1, 1] = -1

    flip = np.linalg.inv(trans) @ tmp @ trans

    return flip


def rot90(image):
    alpha = 90
    rad = 2 * np.pi / 360 * alpha
    trans = np.eye(3)
    trans[0, 2] = -np.array(image).shape[1] / 2
    trans[1, 2] = -np.array(image).shape[0] / 2

    tmp = np.eye(3)
    tmp[0, 0] = np.cos(rad)
    tmp[0, 1] = -np.sin(rad)
    tmp[1, 0] = np.sin(rad)
    tmp[1, 1] = np.cos(rad)
    rot = np.linalg.inv(trans) @ tmp @ trans

    return rot


def main(args):
    max_size = args.max_size
    max_size = max(max_size, 256)
    gene_path = args.gene_image
    stain_path = args.stain_image

    staining_img, gene_img = load_images(gene_path, stain_path)
    img = Normalize(staining_img.astype(np.float32))

    if max_size is not None:
        h, w = img.shape
        if np.max((h, w)) > max_size:
            factor = max_size / np.max((h, w))
            img = zoom(gaussian(img, sigma=3, preserve_range=True), factor, order=1)
            gene = zoom(gaussian(gene_img, sigma=3, preserve_range=True), factor, order=1)
            scaling_matrix = np.eye(3)
            scaling_matrix[0, 0] = factor
            scaling_matrix[1, 1] = factor
        else:
            scaling_matrix = np.eye(3)

    total_matrix = np.eye(3)
    if args.flip_h:
        flip = flip_horizontal(img)
        total_matrix = flip @ total_matrix
    if args.flip_v == 1:
        flip = flip_vertical(img)
        total_matrix = flip @ total_matrix

    for i in range(args.rot90):
        rot = rot90(img)
        total_matrix = rot @ total_matrix

    R = np.linalg.inv(scaling_matrix) @ total_matrix @ scaling_matrix
    out = warpAffine(staining_img, R[:2], (staining_img.shape[1], staining_img.shape[0]))

    total_matrix = intensity_based(args, img, gene, total_matrix.copy())
    try:
        total_matrix = line_based(staining_img, gene_img, total_matrix.copy(), scaling_matrix)
    except:
        print('Skipping line-based registration')
    R = np.linalg.inv(scaling_matrix) @ total_matrix @ scaling_matrix
    out = warpAffine(staining_img, R[:2], (staining_img.shape[1], staining_img.shape[0]))

    gene_out_cropped, stain_out_crooped, crop_params = crop(gene_img, out)

    transformation_parameters = {'affine_matrix': R, 'size': (staining_img.shape[1], staining_img.shape[0]), 'crop_params': crop_params}

    if not os.path.isabs(args.out_dir):
        cwd = os.getcwd()
        out_dir = os.path.join(cwd, args.out_dir)
    else:
        out_dir = args.out_dir
    ensure_dir(out_dir)

    try:
        print(f'Saving results to {out_dir}')
        imwrite(out_dir + '/staining_image_registered.tif', stain_out_crooped)
        imwrite(out_dir + '/gene_image.tif', gene_out_cropped)
        np.save(out_dir + '/parameters.npy', transformation_parameters)
    except:
        print('Saving failed...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gene_image', type=str, required=True)
    parser.add_argument('--stain_image', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_size', type=int, default=512)
    parser.add_argument('--loss', type=str, default='MSE')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--niter', type=float, default=1000)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--flip_h', type=int, default=0)
    parser.add_argument('--flip_v', type=int, default=0)
    parser.add_argument('--rot90', type=int, default=0)

    args = parser.parse_args()

    main(args)
