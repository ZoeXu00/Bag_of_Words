import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses


def get_visual_words(I, dictionary, filterBank):
    filterResponses = extract_filter_responses(I, filterBank)
    pixelResponses = filterResponses.reshape((-1, filterResponses.shape[2])) 
    #In pixelResponses: each pixels responses for each row, total have m*n (size of I) rows

    distances = cdist(pixelResponses, dictionary, metric='euclidean')
    #In distance: row i col j represents distance of pixel i and cluster j
    closestWordIndices = np.argmin(distances, axis=1) #index of cluster picked for each row
    wordMap = closestWordIndices.reshape(I.shape[:2])

    return wordMap
