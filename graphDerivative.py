import numpy as np
import matplotlib.pyplot as plt
from utils import is_outlier, npDerivativeData, manualDerivativeData, \
    load_image, rotateCoords


class GDerivative:
    """
    Graph the derivative of a function given a filepath to an image

    Requirements: Image must have white background with function graphed with any other color
    """

    def __init__(self, imagePath, derivativeFunc="numpy", stripOutliers=True):
        imageData = load_image(imagePath)
        listRGBPixels = list(np.where(imageData != 255))
        xList, yList = rotateCoords(listRGBPixels[0], listRGBPixels[1], -90)

        # Define key graphing variables
        self.fig, self.ax = plt.subplots(2)
        plt.subplots_adjust(hspace=0.6)

        # Plot the original & derivative graph
        self.scatterGraph(0, 'Initial Function', xList, yList, 'red')
        derivativeX, derivativeY = self.computeDerivative(xList, yList, stripOutliers, derivativeFunc)
        self.plotGraph(1, 'Derivative Function', derivativeX, derivativeY, 'blue')

    def computeDerivative(self, listX, listY, stripOutliers, derFunc):
        if derFunc == "manual":
            derivativeX, derivativeY = manualDerivativeData(listX, listY)

            # The manual derivative function does not convert the lists into np arrays
            derivativeX, derivativeY = np.array(derivativeX), np.array(derivativeY)
        else:
            derivativeX, derivativeY = npDerivativeData(listX, listY)

        if stripOutliers:
            # "True" refers to an outlier, "False" refers to not an outlier
            outlierMatrix = is_outlier(derivativeY)

            # Remove the outliers from derivative coords
            derivativeX = derivativeX[outlierMatrix == False]
            derivativeY = derivativeY[outlierMatrix == False]

        # Sort the derivatives as matplotlib requires it for better graphing
        listNewCoords = list(zip(derivativeX, derivativeY))
        listNewCoords.sort(key=lambda tup: tup[0])

        return list(map(list, zip(*listNewCoords)))

    def scatterGraph(self, graphIndex, graphTitle, xCoords, yCoords, colorGraph):
        plt.subplots_adjust(hspace=0.6)
        self.ax[graphIndex].scatter(xCoords, yCoords, color=colorGraph)
        self.ax[graphIndex].set_title(graphTitle)
        self.ax[graphIndex].set_yticklabels([])
        self.ax[graphIndex].set_xticklabels([])

    def plotGraph(self, graphIndex, graphTitle, xCoords, yCoords, colorGraph):
        plt.subplots_adjust(hspace=0.6)
        self.ax[graphIndex].plot(xCoords, yCoords, color=colorGraph)
        self.ax[graphIndex].set_title(graphTitle)
        self.ax[graphIndex].set_yticklabels([])
        self.ax[graphIndex].set_xticklabels([])

    def showGraph(self):
        plt.show()

    def saveGraph(self, fileName):
        self.fig.savefig(fileName, dpi=self.fig.dpi)
