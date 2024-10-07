# Scene Recognition with Bag of Words

## Overview

This project focuses on scene recognition using the **Bag of Words** model. The goal is to classify images into one of eight categories using a subset of the SUN Image database. The approach involves extracting visual words from images, building a visual word dictionary, and implementing nearest neighbor and SVM classifiers.

## Dataset

The dataset consists of **1491 images** from the **SUN Image database**, divided into eight categories. The provided data includes:

- `data/`: Contains the image files categorized into folders.
- `traintest.pkl`: A Python pickle file that stores training and testing image paths along with labels.

## Project Structure
```
SceneRecognition/ │ ├── python/ │ ├── RGB2Lab.py # Provided script for RGB to CIE Lab conversion. │ ├── batchToVisualWords.py # Applies visual words mapping to all images. │ ├── buildRecognitionSystem.py # Builds recognition system and saves the trained model. │ ├── createFilterBank.py # Provided script to generate a set of image filters. │ ├── dictionaryHarris.pkl # Visual words dictionary using Harris corners. │ ├── dictionaryRandom.pkl # Visual words dictionary using random points. │ ├── evaluateRecognitionSystem_NN.py # Evaluates recognition system using nearest neighbors. │ ├── evaluateRecognitionSystem_kNN.py # Evaluates recognition system using k-NN. │ ├── extractFilterResponses.py # Extracts filter responses from an image. │ ├── getDictionary.py # Creates the visual words dictionary. │ ├── getHarrisPoints.py # Detects Harris corner points in an image. │ ├── getImageDistance.py # Computes distance between image histograms. │ ├── getImageFeatures.py # Extracts image histogram of visual words. │ ├── getRandomPoints.py # Selects random points from an image. │ ├── getVisualWords.py # Maps image pixels to their closest visual word. │ ├── utils.py # Helper functions for various operations. │ ├── visionHarris.pkl # Trained recognition model using Harris dictionary. │ ├── visionRandom.pkl # Trained recognition model using random dictionary. │ └── computeIDF.py # Computes inverse document frequency (IDF) (extra credit). │ ├── ec/ │ ├── evaluateRecognitionSystem_IDF.py # Evaluates the system using IDF-weighted recognition. │ ├── evaluateRecognitionSystem_SVM.py # Evaluates the system using SVM (extra credit). │ ├── tryBetterFeatures.py # Experiments with additional image features (extra credit). │ └── any_other_helpers.py # Additional helper scripts. │ ├── <AndrewID>.pdf # Write-up containing answers and results. ├── <AndrewID>.zip # Submission zip file as required for Gradescope. └── README.md # This readme file.
```

