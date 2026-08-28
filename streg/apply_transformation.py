import numpy as np
import argparse

from tifffile import imread, imwrite
from cv2 import warpAffine


def main(args):
    image_path = args.image
    parameters = args.parameters
    out_dir = args.out

    parameters_dict = np.load(parameters, allow_pickle=True).item()

    matrix = parameters_dict['affine_matrix']
    size = parameters_dict['size']
    crop_params = parameters_dict['crop_params']

    img = imread(image_path)

    h, w = img.shape

    if h < size[1]:
        img = np.pad(img, ((0, size[1] - h), (0, 0)))
    if w < size[0]:
        img = np.pad(img, ((0, 0), (size[0] - w)))

    out = warpAffine(img, matrix[:2], size)

    out = out[crop_params[0]:crop_params[1], crop_params[2]:crop_params[3]]

    imwrite(out_dir, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--parameters', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()

    main(args)
