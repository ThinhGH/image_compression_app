import cv2
import sys
import numpy as np
from jpeg_codec import compress_rgb, decompress_rgb
from metrics import psnr, ssim


if len(sys.argv) < 2:
    print("Usage: python main.py input_image")
    sys.exit()

img = cv2.imread(sys.argv[1])
if img is None:
    print("Cannot read image")
    sys.exit()

Yq, Cbq, Crq = compress_rgb(img)

np.save("Y.npy", Yq)
np.save("Cb.npy", Cbq)
np.save("Cr.npy", Crq)

recon = decompress_rgb(Yq, Cbq, Crq, img.shape[:2])
cv2.imwrite("reconstructed_rgb.png", recon)

print("Done!")
print("- Y.npy, Cb.npy, Cr.npy")
print("- reconstructed_rgb.png")
psnr_val = psnr(img, recon)
ssim_val = ssim(img, recon)

print(f"PSNR: {psnr_val:.2f} dB")
print(f"SSIM: {ssim_val:.4f}")
