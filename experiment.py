import cv2
import numpy as np
import matplotlib.pyplot as plt

from jpeg_codec import compress_rgb, decompress_rgb
from metrics import psnr

def quality_psnr_curve(img_path, qualities=range(5, 101, 5)):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    psnr_values = []

    for q in qualities:
        Yq, Cbq, Crq, QY, QC = compress_rgb(img, quality=q)
        recon = decompress_rgb(Yq, Cbq, Crq, QY, QC, (h, w))

        p = psnr(img, recon)
        psnr_values.append(p)

        print(f"Quality={q:3d} → PSNR={p:.2f} dB")

    return qualities, psnr_values
def plot_quality_psnr(qualities, psnr_values):
    plt.figure(figsize=(8, 5))
    plt.plot(qualities, psnr_values, marker='o')
    plt.xlabel("JPEG Quality Factor")
    plt.ylabel("PSNR (dB)")
    plt.title("Quality vs PSNR")
    plt.grid(True)
    plt.show()
