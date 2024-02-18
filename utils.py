from numpy import all, any, array, arctan2, cos, sin, exp, dot, log, logical_and, roll, sqrt, stack, trace, \
unravel_index, pi, deg2rad, rad2deg, where, zeros, floor, full, nan, isnan, round, float32
from numpy.linalg import det, lstsq, norm
from cv2 import resize, GaussianBlur, subtract, KeyPoint, INTER_LINEAR, INTER_NEAREST
from functools import cmp_to_key
import logging
logger = logging.getLogger(__name__)
float_tolerance = 1e-7
def getbaseimage(im, std, blurstd):
    """ This function returns baseline image of the image pyramid given original image.
    In:
    im: image
    std: blur std of op image
    blurstd: assumed blur std of ip image
    Out:
    baseim: ip image doubled and Gaussian blurred by std
    """
    doubleim = resize(im, fx=2, fy=2, interpolation=INTER_LINEAR) #linear interpolated doubled size
    stddiff = sqrt(max(std**2 - (2*blurstd)**2, 0.01))
    baseim = GaussianBlur(im, sigmaX=stddiff, sigmaY=stddiff)
    # image with blur of std instead of assumed blurstd
    return baseim
    
def getnumoctaves(imsize):
    """
    Get number of octaves in image pyramid given base imsize.
    Also equal to number of pyramids, each pyramid at half the previous imsize.
    solve k: y/2^k = 1    (minimum 1 pixel left after k times halving)
    k = round(logy/log2) - 1 (here -1 as margin).
    """
    return int(round(log(min(imsize)) / log(2) - 1))


def gausskerstd(std, numintr):
    """
    Get Gaussian kernel std to blur input image.
    Default std, intervals and octaves based on Sec.3 in original SIFT paper by Lowe.
    Default base std = 1.6
    Default num intervals = 3
    Default images per octave = numintr + 3
    gausskerstds = [base, diff1, diff2, diff3, diff4, diff5],
    the diff3 resultant image std = 2xbasestd.
    that is the std used as base for the next halved octave.
    """
    logger.debug("Gaussian kernel stds...")
    numimoct = numintr + 3 # images per octave
    k = 2 ** (1/numintr)
    s = sqrt(k**2-1) * std
    # stds from one blur to the next in Gaussian pyramid
    # first element = std
    # subsequent element = sqrt(newvar - prevvar) = sqrt((k*prevstd)^2 - prevstd^2)
    gausskerstds = array([std] + [s*(k**(id-1)) for id in range(1,numimoct)])
    return gausskerstds
    
    
def getgausspyr(im, numoct, gausskerstds):
    """Generates Gaussian image pyramid given base image,
    number of octaves, stds of each Gaussian kernels."""
    logger.debug("Generating Gaussain pyramid...")
    gausspyr = []
    for octid in range(numoct):
        octpyr = []
        octpyr.append(im) # base image already w/ blur = base std.
        for s in gausskerstds[1:]: #base std already used above.
            im = GaussianBlur(im,sigmaX=s,sigmaY=s)
            octpyr.append(im)
        gausspyr.append(octpyr)
        nextbase = octpyr[-3]
        im = resize(nextbase, (int(nextbase.shape[1]/2), int(nextbase.shape[0]/2)), interpolation=INTER_NEAREST)
    return array(gausspyr)

def getdogpyr(gausspyr):
    """Generate Difference of Gaussian / Laplacian of Gaussian"""
    logger.debug("Generating DoG pyramid...")
    dogpyr = []
    for octpyr in gausspyr:
        dogoct = []
        for im1, im2 in zip(octpyr[:1], octpyr[1:]):
            dogoct.append(subtract(im2, im1))
            # ordinary subtraction won't work as images are unsigned ints.
        dogpyr.append(dogoct)
    return array(dogpyr)