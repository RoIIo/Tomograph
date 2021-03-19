import math
from math import cos, sqrt, sin, ceil, floor

import cv2
from PIL import Image


import scipy
from skimage import io, data
import numpy as np
import pybresenham
import matplotlib.pyplot as plt

from skimage.transform import radon, iradon

detectors = 180  # number of detectors
deltaAlfa = 1  # degrees between (n) and (n+1) emitter
width = 180  # degrees between first and the last detector
file = "D:\Studia\IwM\Tomograf\pictures/0.png"


def load_image(filename):  # Load image

    im = Image.open(file)
    sqrWidth = np.ceil(np.sqrt(im.size[0] * im.size[1])).astype(int)
    im_resize = im.resize((sqrWidth, sqrWidth))
    im_resize.save('output.png')

    image = cv2.imread('output.png', 0)

    return image

def get_positions(alfa,radius):
    emitterX = radius * cos(alfa)
    emitterY = radius * sin(alfa)
    detectorsX = []
    detectorsY = []
    for j in range(detectors):
        detectorsX.append(radius * cos(
            alfa + math.pi - (width / 2) * math.pi / 180 + (j * (width / (detectors - 1))) * math.pi / 180))
        detectorsY.append(radius * sin(
            alfa + math.pi - (width / 2) * math.pi / 180 + (j * (width / (detectors - 1))) * math.pi / 180))
    return emitterX,emitterY, detectorsX, detectorsY


def bresenham(emitterX,emitterY,detectorX,detectorY,side):
    indexes = list(pybresenham.line(emitterX, emitterY, detectorX, detectorY))
    index = []
    for idx, k in enumerate(indexes):
        if -side / 2 <= indexes[idx][1] < side / 2 and -side / 2 <= indexes[idx][0] < side / 2:
            index.append(k)
    return index


def radon_t(image):  # Radon Transformation
    steps = floor(360 / deltaAlfa)
    side, _ = image.shape
    radius = sqrt(2)*side / 2
    sinogram = np.zeros((steps, detectors), dtype=float)
    for i in range(steps):
        print(i)
        # calculate positions of emitter and detectors
        alfa = i * deltaAlfa * math.pi / 180
        emitterX, emitterY, detectorsX,detectorsY = get_positions(alfa,radius)
        # Bresenham method to find pixels which are crossed by line
        for j in range(detectors):
            index = bresenham(emitterX,emitterY,detectorsX[j],detectorsY[j],side)

            for idx, k in enumerate(index):
                sinogram[i][j] += image[index[idx][0]+side//2][index[idx][1]+side//2]

    #maximum = np.max(sinogram)
    # print(maximum)
   # sinogram = np.divide(sinogram, maximum)
    return sinogram


def i_radon_t(sinogram,side):  # Inverse Radon Transformation
    steps = floor(360 / deltaAlfa)
    radius = sqrt(2)*side / 2
    image = np.zeros((side,side), dtype=float)
    for i in range(steps):
        print(i)
        # calculate positions of emitter and detectors
        alfa = i * deltaAlfa * math.pi / 180
        emitterX, emitterY, detectorsX,detectorsY = get_positions(alfa,radius)

        for j in range(detectors):
            index = bresenham(emitterX, emitterY, detectorsX[j], detectorsY[j],side)

            for idx, k in enumerate(index):
                image[index[idx][0]+side//2][index[idx][1]+side//2] += sinogram[i][j]

   # maximum = np.max(image)
   # image = np.divide(image, maximum)
    return image


def start():
    image = load_image(file)
    plt.subplot(211)
    sinogram = radon_t(image)
    plt.imshow(sinogram, cmap=plt.cm.Greys_r)

    plt.subplot(212)
    reconstruction = i_radon_t(sinogram, image.shape[0])
    plt.imshow(reconstruction, cmap=plt.cm.Greys_r)

    plt.show()


if __name__ == '__main__':
    start()
