import numpy as np
import pickle
import cv2 as cv
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features

# -----fill in your implementation here --------
dictionarySize = 500 #change be changed

filterBank = create_filterbank()

with open('../data/traintest.pkl', 'rb') as f:
    train_data = pickle.load(f)

train_imagenames = train_data['train_imagenames']
T = len(train_imagenames)
train_labels = train_data['train_labels'].reshape(-1, 1)

with open('dictionaryRandom.pkl', 'rb') as f:
    dictionary_random = pickle.load(f)

with open('dictionaryHarris.pkl', 'rb') as f:
    dictionary_harris = pickle.load(f)


trainFeatures_random = np.zeros((T, dictionarySize))
trainFeatures_harris = np.zeros((T, dictionarySize))

for i, path in enumerate(train_imagenames):
    print('-- processing %d/%d' % (i, len(train_imagenames)))
    newpath = path.rsplit('.', 1)[0]
    with open('../data/%s_Random.pkl' % newpath, 'rb') as f:
        wordmap_random = pickle.load(f)
    with open('../data/%s_Harris.pkl' % newpath, 'rb') as f:
        wordmap_harris = pickle.load(f)
    # I = cv.imread('../data/%s' % path)
    # I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    # wordmap_random = get_visual_words(I, dictionary_random, filterBank)
    # wordmap_harris = get_visual_words(I, dictionary_harris, filterBank)

    h_random = get_image_features(wordmap_random, dictionarySize)
    h_harris = get_image_features(wordmap_harris, dictionarySize)

    trainFeatures_random[i] = h_random
    trainFeatures_harris[i] = h_harris


dictionaryRandom = dict()
dictionaryRandom['dictionary'] = dictionary_random
dictionaryRandom['filterBank'] = filterBank
dictionaryRandom['trainFeatures'] = trainFeatures_random
dictionaryRandom['trainLabels'] = train_labels

dictionaryHarris = dict()
dictionaryHarris['dictionary'] = dictionary_harris
dictionaryHarris['filterBank'] = filterBank
dictionaryHarris['trainFeatures'] = trainFeatures_harris
dictionaryHarris['trainLabels'] = train_labels

with open('visionRandom.pkl', 'wb') as f:
    pickle.dump(dictionaryRandom, f)

with open('visionHarris.pkl', 'wb') as f:
    pickle.dump(dictionaryHarris, f)

# ----------------------------------------------
