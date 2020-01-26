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
        y = math.ceil(num / 2.0)
        return (x, y)

    def plotImage(self, image):
        plt.imshow(image, cmap='gray', interpolation='bicubic')
        plt.xticks([])
        plt.yticks([])
    
    def showImages(self, *images):
        plt.gcf().clear()
        plt.title("Microstructures")
        xdim, ydim = self.getDimensions(len(images))
        for i, img in enumerate(images):
            plt.subplot(xdim, ydim, i+1)
            self.plotImage(img)
        plt.show()

    def __init__(self, img_num):
        raw = cv.imread("prec_data_flat/image_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        lam = cv.imread("prec_data_flat/lammask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        p1 = cv.imread("prec_data_flat/p1mask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        p2 = cv.imread("prec_data_flat/p2mask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        prec = cv.imread("prec_data_flat/precmask_{}.png".format(img_num), cv.IMREAD_GRAYSCALE)
        self.showImages(raw, lam, p1, p2, prec)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        GrainDetector(sys.argv[1])
    else:
        GrainDetector(0)