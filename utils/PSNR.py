import math
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def compare_frames(img1, img2):
    score_ssim, diff = compare_ssim(img1, img2, full=True, multichannel=True)
    score_psnr = calculate_psnr(img1, img2)
    return score_ssim, score_psnr
