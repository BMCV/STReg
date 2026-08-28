import math
import numpy as np

from scipy.ndimage import gaussian_filter1d, binary_fill_holes, binary_dilation, grey_erosion
from scipy.signal import argrelmin
from scipy.spatial.distance import cdist
from skimage.filters import gaussian
from skimage.transform import probabilistic_hough_line
from cv2 import warpAffine

from streg.utils import SliceBuilder, Normalize


def extract_crossing_points_gene(img, order=200, gaussian_sigma=None):
    sum_x = img.sum(1)
    sum_y = img.sum(0)
    if gaussian_sigma is not None:
        sum_x = gaussian_filter1d(sum_x, sigma=gaussian_sigma)
        sum_y = gaussian_filter1d(sum_y, sigma=gaussian_sigma)

    minima_x = argrelmin(sum_x, order=order)
    minima_y = argrelmin(sum_y, order=order)

    X, Y = np.meshgrid(minima_x[0], minima_y[0], indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel()])
    return coords


def extract_crossing_points_stain(x, patch_size, detection_threshold=0, gaussian_sigma=None):
    p_x = np.min((x.shape[0], patch_size[0]))
    p_y = np.min((x.shape[1], patch_size[1]))

    s_x = p_x
    s_y = p_y

    sb = SliceBuilder(x, (p_x, p_y), (s_x, s_y))
    x_norm = Normalize(x)
    x_smooth = gaussian(x_norm, sigma=10, preserve_range=True)
    mask = x_smooth > detection_threshold

    mask = binary_dilation(mask, iterations=5)
    mask = binary_fill_holes(mask)

    x_in = gaussian(x_norm, sigma=3, preserve_range=True)

    min_v = grey_erosion(x_in, size=(100, 1))
    min_h = grey_erosion(x_in, size=(1, 100))

    tmp = np.zeros_like(x_in)
    tmp += min_v == x_in
    tmp += min_h == x_in

    tmp[x_in == 0] = 0

    edges = tmp > 0

    all_lines = []

    for i, s in enumerate(sb.img_slices):
        current_x = edges[s]
        current_mask = mask[s]

        lines = probabilistic_hough_line(current_x * (1 - current_mask), threshold=100, line_length=500, line_gap=100)

        lines = np.array(lines)
        if len(lines) > 0:
            lines[:, :, 0] += s[1].start
            lines[:, :, 1] += s[0].start

            if len(all_lines) == 0:
                all_lines = lines
            else:
                all_lines = np.concatenate((all_lines, lines), 0)

    crossings = []
    for i in range(len(all_lines)):
        for j in range(i, len(all_lines)):
            if i == j:
                continue

            x1, y1 = all_lines[i][0]
            x2, y2 = all_lines[i][1]
            x3, y3 = all_lines[j][0]
            x4, y4 = all_lines[j][1]

            enum = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4))
            denum = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

            enum2 = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2))
            denum2 = ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))

            if denum != 0 and denum2 != 0:
                t = enum / denum
                u = enum2 / denum2
                if t >= 0 and t <= 1 and u >= 0 and u <= 1:

                    intersection_point = (y1 + t * (y2 - y1), x1 + t * (x2 - x1))

                    if len(crossings) == 0:
                        crossings.append(np.array(intersection_point))
                    else:
                        add = True
                        for c in crossings:
                            if math.isclose(c[0], intersection_point[0], abs_tol=3) and math.isclose(c[1], intersection_point[1], abs_tol=3):
                                c[0] = (c[0] + intersection_point[0]) / 2
                                c[1] = (c[1] + intersection_point[1]) / 2
                                add = False
                        if add:
                            crossings.append(np.array(intersection_point))

    if len(crossings) == 0:
        raise Exception('No crossing points found. Cannot perform trackline-based image registration...')
    crossings = np.stack(crossings)

    return crossings


def find_matching_points(points1, points2, thresh):

    matrix = cdist(points1, points2)
    matrix[matrix > thresh] = np.inf

    p0 = []
    p1 = []

    while (matrix != np.inf).any():
        min_value = np.unravel_index(matrix.argmin(), matrix.shape)
        p0.append(points1[min_value[0]])
        p1.append(points2[min_value[1]])

        matrix[min_value[0], :] = np.inf
        matrix[:, min_value[1]] = np.inf

    if len(p0) > 3:
        print(f'Found {len(p0)} matching crossingpoints...')
    else:
        raise Exception('No matching crossingpoints found. Cannot perform trackline-based image registration...')

    return np.vstack(p0), np.vstack(p1)


def get_affine_matrix(p1, p2):

    A = []
    b = []

    for (x1, y1), (x2, y2) in zip(p1, p2):
        A.append([x1, y1, 1, 0, 0, 0])
        A.append([0, 0, 0, x1, y1, 1])
        b.append(x2)
        b.append(y2)

    A = np.array(A)
    b = np.array(b)

    matrix, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return np.vstack([matrix.reshape(2, 3), [0, 0, 1]])


def RANSAC(p1, p2, iterations=1000, thresh=2):
    print('Performing RANSAC...')
    best_inliers = []
    best_matrix = []

    N = p1.shape[0]

    for it in range(iterations):
        indices = np.random.choice(N, 3, replace=False)
        p1_sample = p1[indices]
        p2_sample = p2[indices]

        affine_matrix = get_affine_matrix(p1_sample, p2_sample)

        p1_extended = np.hstack([p1, np.ones((p1.shape[0], 1))])
        p1_transformed = (affine_matrix @ p1_extended.T).T[:, :2]

        error = np.linalg.norm(p1_transformed - p2, axis=1)
        inliers = np.where(error < thresh)[0]
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_matrix = affine_matrix

    print(f'Best transformation matrix yields {len(best_inliers)} inliers...')
    p1_final = p1[best_inliers]
    p2_final = p2[best_inliers]

    best_matrix = get_affine_matrix(p1_final, p2_final)
    return best_matrix


class Line_Registration():
    def __init__(self,
                 patch_size=(2000, 2000),
                 stride=(2000, 2000),
                 detection_threshold=0.05,
                 order=200,
                 matching_threshold=100,
                 RANSAC_iterations=5000,
                 RANSAC_threshold=1,
                 gaussian_sigma=None):

        self.patch_size = patch_size
        self.stride = stride
        self.detection_threshold = detection_threshold
        self.order = order
        self.matching_threshold = matching_threshold
        self.RANSAC_iterations = RANSAC_iterations
        self.RANSAC_threshold = RANSAC_threshold
        self.gaussian_sigma = gaussian_sigma
        self.registration_matrix = None

    def optmize(self, staining, genes):
        print('Start trackline registration...')

        coords_gene = extract_crossing_points_gene(genes, order=self.order, gaussian_sigma=self.gaussian_sigma)

        coords_staining = extract_crossing_points_stain(
            staining,
            patch_size=self.patch_size,
            detection_threshold=self.detection_threshold,
            gaussian_sigma=self.gaussian_sigma
        )

        coords_gene_, coords_staining_ = find_matching_points(coords_gene, coords_staining, thresh=self.matching_threshold)
        matrix = RANSAC(coords_staining_, coords_gene_, iterations=self.RANSAC_iterations, thresh=self.RANSAC_threshold)

        self.registration_matrix = matrix

    def apply_registration(self, img):
        p = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, 1]], dtype=float)

        tmp = p @ self.registration_matrix @ p

        return warpAffine(img, tmp[:2], (img.shape[1], img.shape[0]))
