import numpy as np
import cv2

from utils import block_process, pad_to_block
from entropy import *

BLOCK = 8

# ================== Quantization Tables ==================

QY = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
])

QC = np.array([
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
])

# ================== Utilities ==================

def scale_quant_table(Q, quality):
    quality = max(1, min(quality, 100))
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    Qs = np.floor((Q * scale + 50) / 100)
    Qs[Qs == 0] = 1
    return Qs

def dct2(block):
    return cv2.dct(block.astype(np.float32) - 128)

def idct2(block):
    return cv2.idct(block) + 128

# ================== Core JPEG blocks ==================

def compress_channel(channel, Q):
    channel = pad_to_block(channel, BLOCK)
    dct_img = block_process(channel, BLOCK, dct2)
    return block_process(dct_img, BLOCK, lambda b: np.round(b / Q))

def decompress_channel(q_channel, Q):
    dequant = block_process(q_channel, BLOCK, lambda b: b * Q)
    img = block_process(dequant, BLOCK, idct2)
    return np.clip(img, 0, 255)

# ================== Zigzag + RLE ==================

def encode_block(q_block):
    """
    q_block: 8x8 quantized DCT block
    return: [DC, (run, val), ..., (0,0)]
    """
    zz = zigzag(q_block)
    dc = int(zz[0])
    ac = zz[1:]
    rle = rle_encode(ac)
    return [dc] + rle

# ================== Entropy coding (Huffman) ==================

def entropy_encode(symbol_blocks):
    """
    symbol_blocks: list of blocks, each block is [DC, (run,val), ...]
    """
    flat = []
    block_sizes = []

    for blk in symbol_blocks:
        block_sizes.append(len(blk))
        flat.extend(blk)

    tree = build_tree(flat)
    book = build_codebook(tree)
    bits = encode_bits(flat, book)

    meta = {
        "tree": tree,
        "block_sizes": block_sizes
    }

    return bits, meta

# ================== Main JPEG-like Pipeline ==================

def compress_rgb(img_bgr, quality=50, enable_entropy=False):
    """
    Returns:
    Yq, Cbq, Crq, QY, QC, info
    """
    H, W = img_bgr.shape[:2]

    # RGB → YCrCb
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # 4:2:0 subsampling
    Cb_sub = cv2.resize(Cb, (Cb.shape[1]//2, Cb.shape[0]//2))
    Cr_sub = cv2.resize(Cr, (Cr.shape[1]//2, Cr.shape[0]//2))

    # Quant tables
    QY_s = scale_quant_table(QY, quality)
    QC_s = scale_quant_table(QC, quality)

    # DCT + Quant
    Yq  = compress_channel(Y,  QY_s)
    Cbq = compress_channel(Cb_sub, QC_s)
    Crq = compress_channel(Cr_sub, QC_s)

    # ================== Bitrate estimation ==================

    if enable_entropy:
        symbol_blocks = []

        for ch in [Yq, Cbq, Crq]:
            h, w = ch.shape
            for i in range(0, h, BLOCK):
                for j in range(0, w, BLOCK):
                    symbol_blocks.append(
                        encode_block(ch[i:i+BLOCK, j:j+BLOCK])
                    )

        bits, _ = entropy_encode(symbol_blocks)
        total_bits = len(bits)
    else:
        total_bits = (Yq.size + Cbq.size + Crq.size) * 8

    bpp = total_bits / (H * W)

    info = {
        "bits": total_bits,
        "bpp": bpp
    }

    return Yq, Cbq, Crq, QY_s, QC_s, info

# ================== Decompression ==================

def decompress_rgb(Yq, Cbq, Crq, QY, QC, shape):
    Y  = decompress_channel(Yq,  QY)
    Cb = decompress_channel(Cbq, QC)
    Cr = decompress_channel(Crq, QC)

    Y = Y[:shape[0], :shape[1]]
    Cb = cv2.resize(Cb, (shape[1], shape[0]))
    Cr = cv2.resize(Cr, (shape[1], shape[0]))

    ycrcb = cv2.merge([
        Y.astype(np.uint8),
        Cr.astype(np.uint8),
        Cb.astype(np.uint8)
    ])

    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
