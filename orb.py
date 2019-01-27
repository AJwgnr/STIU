import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage import exposure
from skimage import feature
# For iterating through folders
import glob
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set()
import pickle
from sklearn import neighbors
from sklearn.metrics.pairwise import euclidean_distances

os.chdir('C:/Users/Alexander/master/PictureClassification/res/dataset')

#%%
## General stuff
# Total amount of images
def main():
    nr_of_images = 9143
    # Create vector containing the category information
    category_list = os.listdir("101_ObjectCategories/.")
    categs = ["" for x in range(nr_of_images)]
    categoryCount = 0
    categoryNumber = 0
    imagesPerCategroy = []
    i = 0
    
    # categs is a vector that tells the category of each image
    for category in category_list:
        for filename in glob.glob('101_ObjectCategories/'+category+'/*.jpg'):
            categs[i] = category
            categoryCount = categoryCount+1
            i = i+1
            
        imagesPerCategroy.append(categoryCount)
        categoryCount = 0
        categoryNumber = categoryNumber +1 
    

    ind = np.arange(len(imagesPerCategroy))    # the x locations for the groups
    width = 0.5       # the width of the bars: can also be len(x) sequence
    p1 = plt.bar(ind, imagesPerCategroy, width, color='#d62728')    
    #plt.xticks(ind,category_list)
    plt.show()
   
    
    # Initiate STAR detector with limited amount of features
    orb = cv2.ORB_create(nfeatures=100)
    maxFeat = 0
        
    for category in category_list:
        for filename in glob.glob('101_ObjectCategories/'+category+'/*.jpg'):
            # Read image as grayscale, therefor argument 0
            img = cv2.imread(filename, 0)
            # Make image 200x200
            img = cv2.resize(img,(200, 200), interpolation = cv2.INTER_CUBIC)
            # find the keypoints with ORB
            kp = orb.detect(img,None)
    
            # compute the descriptors with ORB
            kp, des = orb.compute(img, kp)
            
            
            if des is None:
                print(filename)
                continue
            
            #find longest vector
            if(maxFeat<(len(des.flatten()))):
                maxFeat = (len(des.flatten()))
    
    FeatSpace = np.zeros((nr_of_images, maxFeat))
    print('Featspace shape: ' + str(FeatSpace.shape))
    
    
 
        
    
    i = 0
    for category in category_list:
        for filename in glob.glob('101_ObjectCategories/'+category+'/*.jpg'):
            # Read image as grayscale, therefor argument 0
            img = cv2.imread(filename, 0)
            # Make image 200x200
            img = cv2.resize(img,(200, 200), interpolation = cv2.INTER_CUBIC)
            # find the keypoints with ORB
            
            kp = orb.detect(img,None)
            # compute the descriptors with ORB
            kp, des = orb.compute(img, kp)
            if des is None:
                continue
            # Add feature vector into feature space
            length  = len(des.flatten())
            des = des.reshape([length,])
            #addd feature vector -> with added zeros for same length
            final = np.concatenate([des, np.zeros(maxFeat - length)])
            
            FeatSpace[i,:] = final
            i = i+1
    
            
            
    nr_components = 20
    # Create model
    pca = PCA(n_components=nr_components)
    # Fit model (probably not needed)#
    #pca.fit(FeatSpace)
    # Transform feature space according to model
   # pca_result = pca.fit_transform(FeatSpace)
    #print(pca_result)
    #print('PCa:' + str(pca_result))
   # print(FeatSpace.shape)
   # print(len(categs))
    
    
    # Create SVM model
    sup = svm.SVC(kernel='linear', class_weight="balanced", verbose=True)
    # Create classification pipeline
    clf = make_pipeline(pca, sup)
    print(clf)
    clf.fit(FeatSpace,categs)
    print('done')
#print(len(categs))
# Calculate score via 10-fold cross-validation
#scores = cross_val_score(clf, FeatSpace, categs, cv=10, n_jobs=-1)
# Print mean accuracy
#print(scores.mean()) # -> 0.49888




if __name__ == "__main__":
    main()
 
## Outdated: Combined in cell "Create classification model"
## Calculate 10-fold crossvalidation accuracy
# Create classifier
#clf = svm.SVC(kernel='linear', class_weight="balanced")
#print('svm')
# 10-fold cross validation
#scores = cross_val_score(clf, pca_result, categs, cv=10)
# Print mean accuracy
#print(scores.mean())
