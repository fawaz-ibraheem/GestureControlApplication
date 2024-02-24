# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf

import handshape_feature_extractor
import frameextractor
import random

from sklearn.metrics.pairwise import cosine_similarity
import csv


## import the handfeature extractor class

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video




# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video




# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================


IMAGE_SIZE = 250


def training_images(directory, hfe):
    images = []
    labels = []
    frames_path = 'frames'
    for image in os.listdir(directory):
        image_path = directory+image
        category = image.split(".")[0].split("_")[0]
        if (category == "DecreaseFanSpeed"):
            category = '10'
        if (category == "DecereaseFanSpeed"):
            category = '10'
        if (category == "TurnOffFan"):
            category = '11'
        if (category == "TurnOnFan"):
            category = '12'
        if (category == "IncreaseFanSpeed"):
            category = '13'
        if (category == "TurnOffLight"):
            category = '14'
        if (category == "TurnOnLight"):
            category = '15'
        if (category == "SetThermostatToSpecifiedTemperature"):
            category = '16'
        for i in range(44,56, 2):
            rand = random.randint(0, 100000)
            frameextractor.frameExtractor(image_path, frames_path, i/100, rand)
            img = cv2.imread(frames_path + "/%#05d.png" % (rand+1))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.rotate(img, cv2.ROTATE_180)
            x_training = hfe.extract_feature(img)
            images.append(x_training)
            labels.append(category)
    return np.array(images), np.array(labels)



def test_images(directory, hfe):
    images = []
    labels = []
    frames_path = 'frames-test'
    for image in os.listdir(directory):
        image_path = directory+image
        category = image.split(".")[0].split("-")[2]
        if (category == "DecreaseFanSpeed"):
            category = '10'
        if (category == "DecereaseFanSpeed"):
            category = '10'
        if (category == "FanOff"):
            category = '11'
        if (category == "FanOn"):
            category = '12'
        if (category == "IncreaseFanSpeed"):
            category = '13'
        if (category == "LightOff"):
            category = '14'
        if (category == "LightOn"):
            category = '15'
        if (category == "SetThermo"):
            category = '16'
        rand = random.randint(0, 100000)
        frameextractor.frameExtractor(image_path, frames_path, 1/2, rand)
        img = cv2.imread(frames_path + "/%#05d.png" % (rand+1))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        x_training = hfe.extract_feature(img)
        images.append(x_training)
        labels.append(category)
    return np.array(images), np.array(labels)



def cosine_similarity_classification(test_data, train_data, train_labels):
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    similarities = cosine_similarity(test_data, train_data)
    max_sim_idx = np.argmax(similarities)
    predicted_label = train_labels[max_sim_idx]
    return predicted_label

hfe = handshape_feature_extractor.HandShapeFeatureExtractor()

directory = 'traindata/'
training_images, training_labels = training_images(directory, hfe)


directory = 'test/'
test_images, test_labels  = test_images(directory, hfe)



with open('Results.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(test_images)):
        predicted_label = cosine_similarity_classification(test_images[i], training_images, training_labels)
        row = predicted_label + ',' + test_labels[i]
        writer.writerow([row])

file.close()
