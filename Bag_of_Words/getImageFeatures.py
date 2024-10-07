import numpy as np


def get_image_features(wordMap, dictionarySize):
    wordMap_flattened = wordMap.flatten() # change to row vector
    
    h = np.histogram(wordMap_flattened, bins=range(dictionarySize+1), density = True)[0]
    # h = h.reshape(1, -1) #2d 1*K array
    # print(len(h[0]))
    assert(len(h) == dictionarySize)
    
    return h
