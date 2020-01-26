import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import sys

class GrainDetector():

    # Returns rows and cols to use in the plotting window (based on number of images to display)
    # Tries to make the shape as 'square' as possible
    def getDimensions(self, num):
        x = math.floor(math.sqrt(num))
        y = math.ceil(num / float(x))
        return (x, y)

    def plotImage(self, image):
        plt.imshow(image, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
    
    def plotImages(self, *images):
        plt.gcf().clear()
        plt.title("Microstructures")
        xdim, ydim = self.getDimensions(len(images))
        for i, img in enumerate(images):
            plt.subplot(xdim, ydim, i+1)
            self.plotImage(img)

    # Isolate darker colored grains (solid)
    # Input is raw image
    def getMaterialP1(self, raw_img):
        raw_img = cv.medianBlur(raw_img,11)
        thresh_img = cv.adaptiveThreshold(raw_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 399, 0)
        return thresh_img

    # Isolate lighter colored grains (solid)
    # Input is raw image
    def getMaterialP2(self, raw_img):
        raw_img = cv.medianBlur(raw_img,15)
        thresh_img = cv.adaptiveThreshold(raw_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 399, 0)
        return thresh_img


    def __init__(self, img_num):
        raw = cv.imread("prec_data_flat/image_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)

        # Test Masks
        lam = cv.imread("prec_data_flat/lammask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        p1 = cv.imread("prec_data_flat/p1mask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        p2 = cv.imread("prec_data_flat/p2mask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        prec = cv.imread("prec_data_flat/precmask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        
        # Calculated Masks
        p1_calc = self.getMaterialP1(raw)
        p2_calc = self.getMaterialP2(raw)

        # Plot
        plt.figure(1)
        self.plotImages(raw, lam, p1, p2, prec)
        plt.figure(2)
        self.plotImages(raw, p1_calc, p2_calc)
        plt.show()

        # breakpoint()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        GrainDetector(sys.argv[1])
    else:
        GrainDetector(0)