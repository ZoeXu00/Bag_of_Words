import numpy as np
import pickle
from scipy.stats import mode
from getImageDistance import get_image_distance
import matplotlib.pyplot as plt

with open('../data/traintest.pkl', 'rb') as f:
    data = pickle.load(f)

test_imagenames = data['test_imagenames']
test_labels = data['test_labels'].astype(int)
train_labels = data['train_labels'].astype(int)

with open('visionRandom.pkl', 'rb') as f:
    dictionary = pickle.load(f)
trainFeatures = dictionary['trainFeatures']

with open('testFeatures.pkl', 'rb') as f:
    testF = pickle.load(f)
testFeatures = testF['testFeatures_random']


def applyKNN(test_feature, train_features, method, k):
    dist = np.array([get_image_distance(test_feature, train_feature, method) 
                     for train_feature in train_features])
    k_indices = dist.argsort()[:k]
    k_labels = np.array(train_labels)[k_indices]
    prediction, _ = mode(k_labels)

    return prediction

print('Training.....')
accuracies = []
for k in range(1, 41):
    print('-- processing %d/40' % k)
    predictions = [applyKNN(test_feature, trainFeatures, 'chi2', k) for test_feature in testFeatures]
    accuracy = np.mean(np.array(predictions) == test_labels)
    accuracies.append(accuracy)

print(accuracies)
k_values = list(range(1, 41))
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Accuracy vs. K value for Random & chi2')
plt.xlabel('K value')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(k_values)
plt.show()