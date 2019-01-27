# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 12:15:03 2018

@author: benedikt
"""

import os
import rope # For autocomplete of local variables in spyder
import numpy as np
import cv2
from skimage import feature
# For iterating through folders
import glob
from sklearn.decomposition import PCA
from sklearn import svm
#from sklearn.model_selection import cross_val_score
#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold
#from scipy.fftpack import fft
import seaborn as sns; sns.set()
#import pickle
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
        
#%%
# Create Feature Space and perform HoG feature extraction
# 2916 is the length of one HoG feature vector
FeatSpaceHog = np.zeros((nr_of_images, 2916))
i = 0
for category in category_list:
    for filename in glob.glob('101_ObjectCategories/'+category+'/*.jpg'):
        # Read image as grayscale, therefor argument 0
        img = cv2.imread(filename, 0)
        # Make image 200x200
        img = cv2.resize(img,(200, 200), interpolation = cv2.INTER_CUBIC)
        # Extract Histogram of Oriented Gradients from the image
        H = feature.hog(img, orientations=9, pixels_per_cell=(25, 25), 
                        cells_per_block=(3, 3), transform_sqrt=True, 
                        block_norm="L1", visualise=False)
        # Add feature vector into feature space
        FeatSpaceHog[i,:] = H
        i = i+1

#%%
# Prepare for SE KROSSWÄLIDÄISCHON
groups = np.zeros((nr_of_images))

i = 0
for category in category_list:
    DIR = os.getcwd() + '/101_ObjectCategories/' + category
    nr_imgs = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    small_folds_nr = math.floor(nr_imgs/10)
    for k in range(1, 10):
        for j in range(i, i+small_folds_nr):
            groups[j] = k
            i += 1
    for j in range(i, i + nr_imgs - 9*small_folds_nr):
        groups[j] = 10
        i += 1
 
#%%
gkf = GroupKFold(n_splits=10) 
# nr_components is the number of resulting dimensions after PCA
accuracies = np.zeros((10))
nr_components = 30
# Fold number
k = 0
class_accuracies = np.zeros((102,10))
cnf_matrix = np.zeros((102,102))
for train, test in gkf.split(FeatSpaceHog, categs, groups=groups):
    FeatSpaceTrain = FeatSpaceHog[train]
    FeatSpaceTest = FeatSpaceHog[test]
    categsTrain = np.array(categs)[train]
    categsTest = np.array(categs)[test]
    
    # Create PCA model
    pca = PCA(n_components=nr_components)
    # Transform feature space according to model
    pca_result = pca.fit_transform(FeatSpaceTrain)
    
    # Calculate how far each class is scattered across the PCA feature space
    category_covs = [0] * len(category_list)
    i = 0
    for name in category_list:
        # Create boolean vector of the same size as categs which is true for all
        # entries of categs that correspond to the current name
        comp_arr = [x==name for x in np.asarray(categsTrain)]
        # Use the boolean vector to filter the pca result for all images belonging
        # to the current class. On those, calculate the class scatter with the help
        # of the covariance matrix.
        category_covs[i] = np.linalg.det(np.cov(pca_result[comp_arr,], rowvar=0))**(1/nr_components)
        i += 1
    
    # Instead of using balanced class weights, we create a dictionary of our own..
    # It punished classes that are scattered far across the PCA Feature Space.
    # class_weight values will be between 0.5 and 1.0. The higher this value,
    # the more the corresponding class is considered in the SVM optimization.
    class_weight_thresh = 0.5
    class_weight_vals = 1 - class_weight_thresh/max(category_covs) * np.array(category_covs)
    class_weight_dict = dict(zip(category_list, class_weight_vals))
    
    # Create classifier
    clf = svm.LinearSVC(class_weight=class_weight_dict, random_state = 1)
    clf.fit(FeatSpaceTrain, categsTrain)
    
    # For calculating the accuracy
    predicted_labels = clf.predict(FeatSpaceTest)
    
    m = 0
    for cat in category_list:
        # categsTest vs predicted_labels
        indices_pred = np.argwhere(predicted_labels==cat)
        indices_true = np.argwhere(categsTest==cat)
        nr_correct_classif = np.intersect1d(indices_pred, indices_true)
        class_accuracies[m,k] = nr_correct_classif.shape[0]/np.argwhere(categsTest==cat).shape[0]
        m += 1
        
    
    # tot is the total number of correct classifications
    tot = 0
    for l in range(0,len(predicted_labels)):
        if predicted_labels[l]==categsTest[l]:
            tot += 1
            
    accuracies[k] = tot/len(categsTest)
    print(accuracies[k])
    print(k)

    # Compute Confusion Matrix
    cnf_matrix += confusion_matrix(categsTest, predicted_labels, labels = category_list)
    k += 1

#%%
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid('off')
#%%
# For plotting the COnfusion Matrix
cnf_matrix = cnf_matrix.astype(int)
plt.figure(figsize=(16, 16))
plot_confusion_matrix(cnf_matrix, classes=category_list,
                      title='Confusion Matrix')

#%%
# For calculating the quality for each class for each fold
class_stds = np.std(class_accuracies, axis = 1).reshape(102,1)
qualities = np.concatenate((class_accuracies, class_stds), axis = 1)
np.savetxt("qualities.csv", qualities, delimiter=";")
