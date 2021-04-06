import numpy as np
from PIL import Image
from matplotlib import transforms


def is_outlier(points, thresh=3.5):
    """
    SOURCE: https://stackoverflow.com/a/11886564/

    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def manualDerivativeData(arrayX, arrayY):
    """
    Manual Derivation Function (Not optimized for large arrays)

    Computes slope for all elements from 1 to n-1 using standard
    formula (y2-y1)/(x2-x1)
    """
    derY, derX = [], []
    assert len(arrayX) == len(arrayY)
    for i in range(1, len(arrayX) - 1):
        # Avoid duplicates
        if arrayX[i] in derX:
            continue
        ySub = arrayY[i + 1] - arrayY[i - 1]
        XSub = arrayX[i + 1] - arrayX[i - 1]
        if XSub == 0:
            derY.append(None)
        else:
            computedSlope = ySub / XSub
            derY.append(computedSlope)
        derX.append(arrayX[i])
    return derX, derY


def npDerivativeData(arrayX, arrayY):
    """
    Numpy Based Derivation Function (Optimized for large arrays)

    Computes slope for all elements using np.diff and returns
    an numpy array of size n-1
    """

    derY = np.diff(arrayY) / np.diff(arrayX)
    arrayX = np.array(arrayX)
    xPoints = (arrayX[:-1] + arrayX[1:]) / 2
    return xPoints, derY


def load_image(inpFilename):
    """
    Load an image given an input file name

    Returns the pixels of the image as a numpy array, allowing
    for computations and analysis
    """
    imageO = Image.open(inpFilename)
    imageO.load()
    data = np.asarray(imageO, dtype="int32")
    return data


def rotateCoords(arrayX, arrayY, angleDegrees):
    """
    Performs a rotation of two arrays given input in degrees

    Specifically, uses Affine2D rotation on arrays and returns
    X and Y arrays
    """
    rotFunction = transforms.Affine2D().rotate_deg(angleDegrees)

    # A column stack is required by the rotation function
    coords = np.column_stack((arrayX, arrayY)).astype(float, copy=False)
    coords = rotFunction.transform(coords)

    # Convert tuple output into list and return xList, yList
    coords = set(map(tuple, coords))
    return map(list, zip(*coords))


def ensureNpArray(inputList):
    return isinstance(inputList, np.ndarray)
