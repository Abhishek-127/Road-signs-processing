import numpy as np
# from scipy.constants.constants import pi
# from numpy.ma.core import exp
# import math
# import scipy.ndimage as nd
# import pylab
import PIL
# import matplotlib.mlab as mlab
# import matplotlib.pyplot as plt
# import matplotlib
import sys
from PIL import  Image
# from scipy.misc import toimage
# import scipy.misc
import time
import cv2
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# from matplotlib import colors
# from matplotlib.colors import hsv_to_rgb
#Function to read in a color image
#Returns an array of matrixes indicating RGB values for each pixel
def imread_colour(fname):
    img = cv2.imread(fname)
    imgRGB = np.asarray(img)
    imCr = imgRGB[:,:,0]
    imCg = imgRGB[:,:,1]
    imCb = imgRGB[:,:,2]
    return img,imCr, imCg, imCb

#Function to determine the greatest value given RGB values of a pixel
def getMax(R,G,B):
    if R > G:
        temp = R
    else:
        temp = G
    if B > temp:
        temp = B
    return temp

#Function to determine the smallest value given RGB values of a pixel
def getMin(R,G,B):
    if R < G:
        temp = R
    else:
        temp = G
    if B < temp:
        temp = B
    return temp


def getRedSign(fname):
    # to run code in terminal:
    # python getRGBVals.py imagename.jpg
    #fname = sys.argv[1]
    image = cv2.imread(fname)
    #image = imread_colour(sys.argv[1])
    blur = cv2.GaussianBlur(image, (21, 21), 0)
    #blur = image
    lower = [0, 100,100] #bgr
    upper = [20, 255, 255]

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
   # keep this tagged.
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('mask for hsv spectrum', mask)
    cv2.waitKey(0)
    result = cv2.bitwise_and(image, image, mask=mask)
    red = cv2.countNonZero(mask)
    s = mask.shape[0] * mask.shape[1]
    per = float(red)/float(s)
    per = per * 100

    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('HSV result', result)
    cv2.waitKey(0)
    output = cv2.bitwise_and(image, hsv, mask=mask)
    out = cv2.countNonZero(output.flatten())
    size = image.shape[0] * image.shape[1]
    percentage = (float(out)/float(size)) * 100 
    cv2.imwrite('outputRED.jpg', output)

def getYellowSign(fname):
    # to run code in terminal:
    # python getRGBVals.py imagename.jpg
    image = cv2.imread(fname)
    #image = imread_colour(sys.argv[1])
    blur = cv2.GaussianBlur(image, (21, 21), 0)
    #blur = image
    lower = [15,0 ,0] # still need rgb , 
    upper = [36, 255, 255]

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
   # keep this tagged.
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('mask for hsv spectrum', mask)
    cv2.waitKey(0)
    result = cv2.bitwise_and(image, image, mask=mask)
    red = cv2.countNonZero(mask)
    s = mask.shape[0] * mask.shape[1]
    per = float(red)/float(s)
    per = per * 100

    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('HSV result', result)
    cv2.waitKey(0)
    output = cv2.bitwise_and(image, hsv, mask=mask)
    out = cv2.countNonZero(output.flatten())
    size = image.shape[0] * image.shape[1]
    percentage = (float(out)/float(size)) * 100 
    cv2.imwrite('outputYellow.jpg', output)

def getWhiteSign(fname):
    # to run code in terminal:
    # python getRGBVals.py imagename.jpg
    image = cv2.imread(fname)
    #image = imread_colour(sys.argv[1])
    blur = cv2.GaussianBlur(image, (21, 21), 0)
    #blur = image
    lower = [0,0 ,240] # still need rgb , 
    upper = [255, 15, 255]

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
   # keep this tagged.
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('mask for hsv spectrum', mask)
    cv2.waitKey(0)
    result = cv2.bitwise_and(image, image, mask=mask)
    red = cv2.countNonZero(mask)
    s = mask.shape[0] * mask.shape[1]
    per = float(red)/float(s)
    per = per * 100

    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('HSV result', result)
    cv2.waitKey(0)
    output = cv2.bitwise_and(image, hsv, mask=mask)
    out = cv2.countNonZero(output.flatten())
    size = image.shape[0] * image.shape[1]
    percentage = (float(out)/float(size)) * 100 
    cv2.imwrite('outputYellow.jpg', output)

def getBlueSign(fname):
    # to run code in terminal:
    # python getRGBVals.py imagename.jpg
    image = cv2.imread(fname)
    #image = imread_colour(sys.argv[1])
    blur = cv2.GaussianBlur(image, (21, 21), 0)
    #blur = image
    lower = [110,50 ,50] # still need rgb , 
    upper = [130, 255, 255]

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
   # keep this tagged.
    mask = cv2.inRange(hsv, lower, upper)
    cv2.imshow('mask for hsv spectrum', mask)
    cv2.waitKey(0)
    result = cv2.bitwise_and(image, image, mask=mask)
    red = cv2.countNonZero(mask)
    s = mask.shape[0] * mask.shape[1]
    per = float(red)/float(s)
    per = per * 100

    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('HSV result', result)
    cv2.waitKey(0)
    output = cv2.bitwise_and(image, hsv, mask=mask)
    out = cv2.countNonZero(output.flatten())
    size = image.shape[0] * image.shape[1]
    percentage = (float(out)/float(size)) * 100 
    cv2.imwrite('outputYellow.jpg', output)
def main():
    getBlueSign(sys.argv[1])


if __name__ == '__main__':
    main()
