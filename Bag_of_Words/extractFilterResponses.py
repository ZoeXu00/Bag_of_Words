import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *


def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    I_lab = rgb2lab(I)
    h, w, d = I_lab.shape
    assert(d == 3)
    
    num_filters = len(filterBank)
    filter_responses = np.zeros((h, w, 3*num_filters), dtype=np.float32)
    
    for i, filter in enumerate(filterBank):
        for c in range(3):
            response = imfilter(I_lab[:, :, c], filter)
            #normalization
            response = (response - response.min()) / (response.max() - response.min())
            response = (255 * response).astype('uint8')
            filter_responses[:, :, 3 * i + c] = response

    return filter_responses


