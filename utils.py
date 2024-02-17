from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, \
unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging

def getbaseimage(im, std, blurstd):
    # This function returns baseline image of the image pyramid given original image.
    # In:
    # im: image
    # std: blur std of op image
    # blurstd: assumed blur std of ip image
    # Out:
    # baseim: ip image doubled and Gaussian blurred by std
    doubleim = resize(im, fx=2, fy=2, interpolation=INTER_LINEAR) #linear interpolated doubled size
    stddiff = sqrt(max(std**2 - (2*blurstd)**2, 0.01))
    baseim = GaussianBlur(im, stddiff, stddiff)
    # image with blur of std instead of assumed blurstd
    return baseim
    