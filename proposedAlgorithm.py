import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

def get_input_image(path_to_image):
    image = cv2.imread(path_to_image)
    image_clustering(image)
    return image

def image_clustering(image):
    clustered_image = image.copy()
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


get_input_image('yield.jpg')
