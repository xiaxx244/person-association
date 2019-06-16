# coding: utf-8
"""
Implementation of the classic paper by Zhou Wang et. al.: 
Image quality assessment: from error visibility to structural similarity
https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1284395
"""

from __future__ import division
from scipy.ndimage import gaussian_filter
import numpy as np
import cv2




def get_ssim(X, Y):
    """
       Computes the mean structural similarity between two images.
    """
    assert (X.shape == Y.shape), "Image-patche provided have different dimensions"
    nch = 1 if X.ndim==2 else X.shape[-1]
    mssim = []
    for ch in xrange(nch):
        Xc, Yc = X[...,ch].astype(np.float64), Y[...,ch].astype(np.float64)
        mssim.append(compute_ssim(Xc, Yc))
    return np.mean(mssim)


def compute_ssim(X, Y):
    """
       Compute the structural similarity per single channel (given two images)
    """
    # variables are initialized as suggested in the paper
    K1 = 0.01
    K2 = 0.03
    sigma = 1.5
    win_size = 5   

    # means
    ux = gaussian_filter(X, sigma)
    uy = gaussian_filter(Y, sigma)

    # variances and covariances
    uxx = gaussian_filter(X * X, sigma)
    uyy = gaussian_filter(Y * Y, sigma)
    uxy = gaussian_filter(X * Y, sigma)

    # normalize by unbiased estimate of std dev 
    N = win_size ** X.ndim
    unbiased_norm = N / (N - 1)  # eq. 4 of the paper
    vx  = (uxx - ux * ux) * unbiased_norm
    vy  = (uyy - uy * uy) * unbiased_norm
    vxy = (uxy - ux * uy) * unbiased_norm

    R = 255
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    # compute SSIM (eq. 13 of the paper)
    sim = (2 * ux * uy + C1) * (2 * vxy + C2)
    D = (ux ** 2 + uy ** 2 + C1) * (vx + vy + C2)
    SSIM = sim/D 
    mssim = SSIM.mean()

    return mssim



IMAGE_WIDTH, IMAGE_HEIGHT = 60,  160
SSIM_THR = 0.3

def skReID(im1, im2):
    assert (im1 is not None), "image not found"
    assert (im2 is not None), "image not found"
    global IMAGE_WIDTH
    global IMAGE_HEIGHT
    global SSIM_THR

    # resize images
    im1 = cv2.resize(im1, (IMAGE_WIDTH, IMAGE_HEIGHT))
    im2 = cv2.resize(im2, (IMAGE_WIDTH, IMAGE_HEIGHT))

    ssims = get_ssim(im1, im2)
    #print ssims
    return ssims, (ssims>SSIM_THR)
            





