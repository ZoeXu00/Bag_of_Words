# Scene Recognition with Bag of Words

## Overview

This project focuses on scene recognition using the **Bag of Words** model. The goal is to classify images into one of eight categories using a subset of the SUN Image database. The approach involves extracting visual words from images, building a visual word dictionary, and implementing nearest neighbor and SVM classifiers.

## Dataset

The dataset consists of **1491 images** from the **SUN Image database**, divided into eight categories. The provided data includes:

- `data/`: Contains the image files categorized into folders.
- `traintest.pkl`: A Python pickle file that stores training and testing image paths along with labels.

## Project Structure
```
Bag_of_Words/
│
├── data4/                           # Contains the dataset from the SUN Image database.
│   ├── airport/
│   ├── auditorium/
│   ├── bedroom/
│   ├── campus/
│   ├── desert/
│   ├── football_stadium/
│   ├── landscape/
│   ├── rainforest/
│   └── traintest.pkl                # Contains training and testing image paths and labels.
│
└── python/                          # Contains Python scripts for different stages of the project.
    ├── batchToVisualWords.py        # Applies visual words mapping to all images.
    ├── buildRecognitionSystem.py   # Builds recognition system and saves the trained model.
    ├── computeDictionary.py         # Creates and saves the visual word dictionary.
    ├── createFilterBank.py          # Provided script to generate a set of image filters.
    ├── evaluateRecognitionSystem_kNN.py  # Evaluates recognition system using k-NN.
    ├── evaluateRecognitionSystem_NN.py   # Evaluates recognition system using nearest neighbors.
    ├── extractFilterResponses.py    # Extracts filter responses from an image.
    ├── getDictionary.py             # Creates the visual words dictionary.
    ├── getHarrisPoints.py           # Detects Harris corner points in an image.
    ├── getImageDistance.py          # Computes distance between image histograms.
    ├── getImageFeatures.py          # Extracts image histogram of visual words.
    ├── getRandomPoints.py           # Selects random points from an image.
    ├── getVisualWords.py            # Maps image pixels to their closest visual word.
    ├── RGB2Lab.py                   # Converts RGB images to CIE Lab color space.
    └── utils.py                     # Helper functions for various operations.
```

