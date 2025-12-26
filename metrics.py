import numpy as np
import cv2
import numpy as np

def psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100

    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)


def ssim(img1, img2):

    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

    sigma1_sq = cv2.GaussianBlur(img1**2, (11, 11), 1.5) - mu1**2
    sigma2_sq = cv2.GaussianBlur(img2**2, (11, 11), 1.5) - mu2**2
    sigma12   = cv2.GaussianBlur(img1*img2, (11, 11), 1.5) - mu1*mu2

    ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / \
               ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()
