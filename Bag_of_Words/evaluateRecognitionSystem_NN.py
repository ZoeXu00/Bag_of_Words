import numpy as np
import pickle
import cv2 as cv
import os
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance

dictionarySize = 500 #change be changed

with open('../data/traintest.pkl', 'rb') as f:
    data = pickle.load(f)

test_imagenames = data['test_imagenames']
test_labels = data['test_labels'].astype(int)
train_labels = data['train_labels'].astype(int)

with open('visionRandom.pkl', 'rb') as f:
    dictionaryRandom = pickle.load(f)

with open('visionHarris.pkl', 'rb') as f:
    dictionaryHarris = pickle.load(f)

trainFeatures_random = dictionaryRandom['trainFeatures']
trainFeatures_harris = dictionaryHarris['trainFeatures']

#Build Test Features------------------------------------------------------
if not os.path.exists('testFeatures.pkl'):
    print('----Buiding Test Features')

    filterBank = create_filterbank()

    with open('dictionaryRandom.pkl', 'rb') as f:
        dictionary_random = pickle.load(f)

    with open('dictionaryHarris.pkl', 'rb') as f:
        dictionary_harris = pickle.load(f)

    testFeatures_random = np.zeros((len(test_imagenames), dictionarySize))
    testFeatures_harris = np.zeros((len(test_imagenames), dictionarySize))

    for i, path in enumerate(test_imagenames):
        print('-- processing %d/%d' % (i, len(test_imagenames)))
        newpath = path.rsplit('.', 1)[0]
        with open('../data/%s_Random.pkl' % newpath, 'rb') as f:
            wordmap_random = pickle.load(f)
        with open('../data/%s_Harris.pkl' % newpath, 'rb') as f:
            wordmap_harris = pickle.load(f)

        h_random = get_image_features(wordmap_random, dictionarySize)
        h_harris = get_image_features(wordmap_harris, dictionarySize)

        testFeatures_random[i] = h_random
        testFeatures_harris[i] = h_harris
    
    testF = dict()
    testF['testFeatures_random'] = testFeatures_random
    testF['testFeatures_harris'] = testFeatures_harris
    with open('testFeatures.pkl', 'wb') as f:
        pickle.dump(testF, f)
    print("Test Features saved.")

else:
    with open('testFeatures.pkl', 'rb') as f:
        testF = pickle.load(f)
    testFeatures_random = testF['testFeatures_random']
    testFeatures_harris = testF['testFeatures_harris']

#Apply NN --------------------------------------------------------------
def applyNN(test_feature, train_features, method):
    dist = np.array([get_image_distance(test_feature, train_feature, method) 
                     for train_feature in train_features])
    index = dist.argmin()
    prediction = train_labels[index]
    return prediction

#Evaluate NN
print('---- Training')
predictions_random_e = [applyNN(test_feature, trainFeatures_random, 'euclidean') for test_feature in testFeatures_random]
predictions_random_c = [applyNN(test_feature, trainFeatures_random, 'chi2') for test_feature in testFeatures_random]
predictions_harris_e = [applyNN(test_feature, trainFeatures_harris, 'euclidean') for test_feature in testFeatures_harris]
predictions_harris_c = [applyNN(test_feature, trainFeatures_harris, 'chi2') for test_feature in testFeatures_harris]

def evaluateNN(predictions):
    accuracy = np.mean(np.array(predictions) == test_labels)

    confusion_matrix = np.zeros((8, 8), dtype=int)
    assert(len(predictions) == len(test_labels))
    for true_label, predict_label in zip(test_labels, predictions):
        confusion_matrix[true_label-1, predict_label-1] += 1
    return accuracy, confusion_matrix

a_r_e, m_r_e = evaluateNN(predictions_random_e)
a_r_c, m_r_c = evaluateNN(predictions_random_c)
a_h_e, m_h_e = evaluateNN(predictions_harris_e)
a_h_c, m_h_c = evaluateNN(predictions_harris_c)

print(f'Random + Euclidean:{a_r_e}\n{m_r_e}')
print(f'Random + chi2:{a_r_c}\n{m_r_c}')
print(f'Harris + Euclidean:{a_h_e}\n{m_h_e}')
print(f'Harris + chi2:{a_h_c}\n{m_h_c}')

