
import torch
import math
from math import exp
import numpy as np
import pandas as pd
import cv2
import pylab
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.stats import pearsonr



IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

''' MAE  '''
def calculate_mae(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]
    # img1 = img1 * 0.068582304 + 0.07493994
    # img2 = img2 * 0.195 + 0.453
    # img1 = img1 * 0.195 + 0.453
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mae = np.mean(abs(img1 - img2))
    if mae == 0:
        return float('inf')
    return mae


''' PSNR '''
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]
    # img1 = img1 * 0.195 + 0.453
    # img2 = img2 * 0.195 + 0.453
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    # print(img1)
    # print(img2)
    mse = np.mean((img1 - img2) ** 2)
    rmse = math.sqrt(mse)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / rmse)


''' SSIM '''
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h - border, border:w - border]
    img2 = img2[border:h - border, border:w - border]
    # img1 = img1 * 0.068582304 + 0.07493994
    # img2 = img2 * 0.195 + 0.453
    # img1 = img1 * 0.195 + 0.453
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] != 1:
            ssims = []
            for i in range(img1.shape[0]):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    # C1 = (0.01 * 255) ** 2
    # C2 = (0.03 * 255) ** 2
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()
