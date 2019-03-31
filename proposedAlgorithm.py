import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import math

def get_input_image(path_to_image):
    image = cv2.imread(path_to_image)
    image_clustering(image)
    gabor_filter(image)
    return image

def image_clustering(image):
    clustered_image = image.copy()
    clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2GRAY)
    Z = clustered_image.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((clustered_image.shape))

    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return clustered_image

def plot_clustered_data(img):
    X = np.random.randint(25,50,(25,2))
    Y = np.random.randint(60,85,(25,2))
    Z = np.vstack((X,Y))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,2,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now separate the data, Note the flatten()
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]

    # Plot the data
    plt.scatter(A[:,0],A[:,1])
    plt.scatter(B[:,0],B[:,1],c = 'r')
    plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    plt.xlabel('Height'),plt.ylabel('Weight')
    plt.show()


def check_aspect_ratio(image):
    height = image.shape[0]
    width = image.shape[1]

    if float(height/width) > 1.9 or float(height/width) < float(1/1.9):
        return False

    return True

def gabor_filter(image):
    height = image.shape[0]
    width = image.shape[1]
    g_kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

    # img = cv2.imread('test.jpg')
    img = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # img = image.copy()
    filtered_img = cv2.filter2D(img, cv2.CV_8UC3, g_kernel)

    cv2.imshow('image', img)
    cv2.imshow('filtered image', filtered_img)

    h, w = g_kernel.shape[:2]
    # g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    g_kernel = cv2.resize(filtered_img, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def gabor2(image):
    kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())
    filtered = cv2.filter2D(image, -1, kernel)
    cv2.imshow('Abhishek', filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

get_input_image('yield.jpg')
