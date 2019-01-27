# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:15:03 2018

@author: benedikt
"""

import os
import numpy as np
# For iterating through folders
import glob
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

os.chdir('C:/Studium/Computervisualistik/ImageUnderstanding/Projekt/STIU')

#%%
## General stuff
# Total amount of images
nr_of_images = 9144
# Create vector containing the category information
category_list = os.listdir("101_ObjectCategories/.")

# categs is a vector that is of the size of the total amount of images and
# tells the category of each image
categs = ["" for x in range(nr_of_images)]
i = 0
for category in category_list:
    for filename in glob.glob('101_ObjectCategories/'+category+'/*.jpg'):
        categs[i] = category
        i = i+1
    
model = VGG16()
intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('fc2').output)
        
#%%
# Create Feature Space and perform HoG feature extraction
# 2916 is the length of one HoG feature vector
FeatSpace = np.zeros((nr_of_images, 4096))
i = 0
for category in category_list:
    for filename in glob.glob('101_ObjectCategories/'+category+'/*.jpg'):
        img = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        img = img_to_array(img)
        # reshape data for the model
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        # prepare the image for the VGG model
        img = preprocess_input(img)
        # Generate output for the trimmed model
        intermediate_output = intermediate_layer_model.predict(img)
        
        FeatSpace[i,:] = np.asarray(intermediate_output,dtype=np.float32)
        i = i+1
        print(i)

#%%
## Create classification model
# Create PCA model
# nr_components is the number of resulting dimensions after PCA
nr_components = 30
pca = PCA(n_components=nr_components)
# Create SVM model
sup = svm.SVC(kernel='linear', class_weight="balanced")
# Create classification pipeline
clf = make_pipeline(pca, sup)
# Die Verwendung von SVM ohne PCA scheint bessere Ergebnisse zu liefern,
# deshalb ist sup das erste Argument und nicht clf
# Calculate score via 10-fold cross-validation
scores = cross_val_score(sup, FeatSpace, categs, cv=10)
# Print mean accuracy
print(scores.mean()) # -> 0.49888