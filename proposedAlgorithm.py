import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt
import math

def get_input_image(path_to_image):
    image = cv2.imread(path_to_image)
    # image_clustering(image)
    # gabor_filter(image)
    # octagon(image)
    # fuzzy_red(image)
    # fuzzy_blue(image)
    # fuzzy_green(image)
    return image

def octagon(image):

    # img = cv2.imread(image, 1)
    img = image
    cv2.imshow('img1',img[:,:,0])

    ret,thresh1 = cv2.threshold(img[:,:,0], 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imshow('thresh1', thresh1)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        print (len(approx))
        if len(approx)==8:
            print ("octagon")
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 6)

    cv2.imshow('sign', img)       
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_clustering(image):
    clustered_image = image.copy()
    # clustered_image = cv2.cvtColor(clustered_image, cv2.COLOR_BGR2LAB)
    Z = clustered_image.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    ret,label,center = cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    print(len(label))
    print(len(center))

    center = np.uint8(center)
    A = Z[label.ravel()==0]
    B = Z[label.ravel()==1]
    C = Z[label.ravel()==2]
    # plt.scatter(A[:,0],A[:,1])
    # plt.scatter(B[:,0],B[:,1],c = 'r')
    # plt.scatter(center[:,0],center[:,1],s = 80,c = 'y', marker = 's')
    # plt.xlabel('Height'),plt.ylabel('Weight')
    # plt.show()

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

def calc_perimeter(image):
    height = image.shape[0]
    width = image.shape[1]
    perimeter = (2*height) + (2*width)
    return perimeter


def find_yellow(image):
    # the below is pairs of yellow and red respectively
    hsv_color_pairs = (
        (np.array([21, 100, 75]), np.array([25, 255, 255])),
        (np.array([1, 75, 75]), np.array([9, 225, 225]))
    )
    # this code is in a for loop and loops over the above HSV color ranges
    mask = cv2.inRange(hsv, colors[0], colors[1])
    out = cv2.bitwise_and(im, im, mask=mask)
    blur = cv2.blur(out, (5, 5), 0)
    imgray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 0, 255, cv2.THRESH_OTSU)

    heir, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def bgr_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def fuzzy_red(image):
    hsv_image = bgr_to_hsv(image)
    new_image = hsv_image.copy()
    height, width = get_dim(image)
    cv2.imshow('hsv image', hsv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for x in range(0, height):
        for y in range(0, width):
            hue = hsv_image[x, y][0]
            sat = hsv_image[x,y][1]
            # 173
            if(hue < 173): 
                new_image[x, y] = [0,0,0]
            if(sat < 150):
                new_image[x,y] = [0,0,0]
            
    cv2.imshow('formatted', new_image)
    cv2.waitKey(0)
    return new_image

def get_dim(image):
    return image.shape[0], image.shape[1]

def fuzzy_blue(image):
    hsv_image = bgr_to_hsv(image)
    new_image = hsv_image.copy()
    height, width = get_dim(image)
    for x in range(0, height):
        for y in range(0, width):
            hue = hsv_image[x, y][0]
            sat = hsv_image[x, y][1]
            if(118 < hue < 134):
                new_image[x, y] = hsv_image[x,y]
            else:
                new_image[x, y] = [0, 0, 0]

            if(178 < sat < 255):
                new_image[x, y] = hsv_image[x, y]
            else:
                new_image[x, y] = [0, 0, 0]
    cv2.imshow('Blue image', new_image)
    cv2.waitKey(0)
    return new_image

def fuzzy_green(image):
    hsv_image = bgr_to_hsv(image)
    new_image = hsv_image.copy()
    height, width = get_dim(image)
    for x in range(0, height):
        for y in range(0, width):
            hue = hsv_image[x, y][0]
            sat = hsv_image[x, y][1]
            if(75 < hue < 91):
                new_image[x, y] = hsv_image[x,y]
            else:
                new_image[x, y] = [0, 0, 0]

            if(178 < sat < 255):
                new_image[x, y] = hsv_image[x, y]
            else:
                new_image[x, y] = [0, 0, 0]
    cv2.imshow('Green image', new_image)
    cv2.waitKey(0)
    return new_image  
            

def main():
    if len(sys.argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1

    img_name = sys.argv[1]
    image = get_input_image(img_name)
    red = fuzzy_red(image)
    cv2.imwrite('circle.png', red)
    return 0


if __name__ == '__main__':
    main()

