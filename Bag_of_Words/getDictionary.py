import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
        # -----fill in your implementation here --------
        responses = extract_filter_responses(image, filterBank)
        H, W, _ = responses.shape
        if method == 'Random':
            pts = get_random_points(image, alpha)
        elif method == 'Harris':
            pts = get_harris_points(image, alpha, k=0.05)
        else:
            raise ValueError("Unknown method. Use 'Random' or 'Harris'.")
        
        assert(len(pts) == alpha)
        assert(responses.shape[:2] == image.shape[:2])

        for j, (x, y) in enumerate(pts):
            pixelResponses[i * alpha + j, :] = responses[x, y, :].flatten()
        # ----------------------------------------------
    print("Start KMeans.....")
    dictionary = KMeans(n_clusters=K, random_state=0, algorithm='elkan').fit(pixelResponses).cluster_centers_
    return dictionary
