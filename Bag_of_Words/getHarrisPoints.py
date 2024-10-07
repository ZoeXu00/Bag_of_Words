import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter


def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0


    Ix = ndimage.sobel(I, 0)
    Iy = ndimage.sobel(I, 1)
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2

    Sxx = ndimage.gaussian_filter(Ixx, sigma=1)
    Sxy = ndimage.gaussian_filter(Ixy, sigma=1)
    Syy = ndimage.gaussian_filter(Iyy, sigma=1)


    det_M = (Sxx * Syy) - (Sxy**2)
    trace_M = Sxx + Syy
    R = det_M - k * (trace_M**2)
    # print(R.shape)

    flat_R = R.flatten()
    indices = np.argsort(-flat_R)
    selected_indices = indices[:alpha]
    points = np.array(np.unravel_index(selected_indices, R.shape)).T
    
    return points

