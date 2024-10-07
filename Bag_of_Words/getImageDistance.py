import numpy as np
from utils import chi2dist

def get_image_distance(hist1, hist2, method):
    if method == 'euclidean':
        dist = np.sqrt(np.sum((hist1 - hist2)**2))
    elif method == 'chi2':
        dist = chi2dist(hist1, hist2)
    else:
        raise ValueError("Unknown method. Use 'euclidean' or 'chi2'.")
    
    return dist
