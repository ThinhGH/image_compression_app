import numpy as np
import cv2

def block_process(img, block_size, func):
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = img[i:i+block_size, j:j+block_size]
            out[i:i+block_size, j:j+block_size] = func(block)

    return out
def pad_to_block(img, block=8):
    h, w = img.shape
    pad_h = (block - h % block) % block
    pad_w = (block - w % block) % block
    return np.pad(
        img,
        ((0, pad_h), (0, pad_w)),
        mode='edge'
    )
