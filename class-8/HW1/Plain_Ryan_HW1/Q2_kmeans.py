
from turtle import distance
import imageio
from matplotlib.pyplot import axis
import numpy as np
from scipy.spatial.distance import cdist
import time
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as img

## Add image path and clusters
IMG_PATH = None
K = None
# eulcidean or cityblock
DISTANCE = None

class Kmean: 

    def __init__(self, img, distance='euclidean'):
        self.img = img
        self.distance = distance
        self.data = self.parse_image(img)


    def parse_image(self, img):
        """
        img: path to image

        return: 2D array of pixels in size (n x m, 3)
        """
        img = imageio.imread(img)
        
        # remove RGB-A format
        if img.shape[2] == 4:
            img = img[:, :, :3]
        
        self.H = img.shape[0]
        self.W = img.shape[1]
        
        return (img/255).reshape(-1, 3)




    def kmeans(self, X, k):
        
        """
        X: the reshaped image, each row is a pixel
        
        k: the number of clusters
        """
        
        self.k = k
        t0 = time.time()
        # Random centroid assignment
        rand_centroid = np.random.choice(len(X), k, replace=False)
        centroids = X[rand_centroid, :]
        
        # If the centroids were not unique from the random assignment,
        # continue to assign until the unique rows == k
        while np.unique(centroids, axis=0).shape[0] != k:
            rand_centroid = np.random.choice(len(X), k, replace=False)
            centroids = X[rand_centroid, :]
        
        tmpdiff = cdist(X, centroids, self.distance)

        labels = np.array([np.argmin(i) for i in tmpdiff])
        
        # Test if the new centroids differ from the previous ones by more than 1e-3
        convergence = False
        iteration = 0
        
        while convergence == False:
            
            
            iteration += 1
            centroids_prev = centroids
            tmp = []
            
            for cluster in range(k):
                
                # Use the mean for euclidian distance
                # or the median for the manhattan distance
                if self.distance == 'euclidean':
                    center = X[labels==cluster].mean(axis=0)
                else:
                    center = np.median(X[labels==cluster], axis=0)
                tmp.append(center)

            centroids = np.vstack(tmp)

            diff = cdist(X, centroids, self.distance)

            labels = np.array([np.argmin(i) for i in diff])

            if abs(centroids - centroids_prev).max() < 0.001:
                convergence = True
            
            t1 = time.time()
            
        
        
        self.labels = labels
        self.centroids = centroids
        self.iteration = iteration
        self.time_conv = round(t1 - t0, 3)
        
        print(f'Iteration to converge for {k} clusters is {iteration} in {(self.time_conv)} secs')


    def compress_img(self, trial=""):
        """
        k: the number of clusters for filename
        
        trial: additional information to add to file name 
        """
        x = [self.centroids[i] for i in self.labels]
        
        data = np.vstack(x).reshape(self.H, self.W, 3)
        
        plt.imshow(data)
        
        plt.imsave(f'{self.img}_{self.k}_cluster_{self.distance}_{self.iteration}_iter_{trial}.png', data)
        
    def attributes(self):
        
        df = pd.DataFrame(self.data)
        df['label'] = self.labels
        label_var = df.groupby('label')[df.drop('label', axis=1).columns].var().sum().sum()

        return {
            'image': self.img,
            'k': self.k,
            'distance': self.distance,
            'centroids': self.centroids * 255,
            'variance': label_var,
            'iteration': self.iteration,
            'time': self.time_conv
        }


if IMG_PATH and K:
    kmean = Kmean(IMG_PATH, DISTANCE)
    kmean.kmeans(kmean.data, K)
    kmean.compress_img()

