# @author: Miriam Snow
# @email: msnow01@uoguelph.ca
# @ID: 0954174
# CIS *4720 Image Processing
# Assignment 2 Fire Detection algorithm

import numpy as np
import PIL
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib
import pylab

#Function to read in a color image
#Returns an array of matrixes indicating RGB values for each pixel
def imread_colour(fname):
    img = PIL.Image.open(fname)
    imgRGB = np.asarray(img)
    imCr = imgRGB[:,:,0]
    imCg = imgRGB[:,:,1]
    imCb = imgRGB[:,:,2]
    return imCr, imCg, imCb

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

#Function to calculate the hue given RGB values of a pixel
def calcHue(R,G,B):
    red = float(R)/255
    green = float(G)/255
    blue = float(B)/255
    max = getMax(R,G,B)
    min = getMin(R,G,B)
    max = float(max)/255
    min = float(min)/255
    if max == 0:
        hue = 0
        return hue
    elif max == red:
       hue = (((green-blue)/(max-min))%6)*60
       return hue
    elif max == green:
        hue = (((blue-red)/(max-min))+2)*60
        return hue
    elif max == blue:
        hue = (((red-green)/(max-min))+4)*60
        return hue

#Function to calculate the value given RGB values of a pixel
def calcVal(R,G,B):
    max = getMax(R,G,B)
    val = (float(max)/255)*100
    return val

#Function to calculate the saturation given RGB values of a pixel
def calcSat(R,G,B):
    max = getMax(R,G,B)
    min = getMin(R,G,B)
    max = float(max)/255
    min = float(min)/255
    if max == 0:
        sat = 0
        return sat
    else:
        sat = ((max-min)/max)*100
        return sat

def main():
    img = imread_colour("fire1.jpg")
    R = img[0][0][0]
    G = img[1][0][0]
    B = img[2][0][0]
    val = calcVal(R,G,B)
    hue = calcHue(R,G,B)
    sat = calcSat(R,G,B)
    print("First pixel of image fire1.jpg")
    print('Hue:' + str(round(hue,2)))
    print('Saturation:' + str(round(sat,2)))
    print('Value:' + str(round(val,2)))

if __name__ == '__main__':
    main()
