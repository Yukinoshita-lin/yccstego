"""8x8 分块 2D-DCT/IDCT、标准量化表及其质量缩放、之字扫描。"""
from __future__ import annotations

import numpy as np

BLOCK = 8


def _dct_matrix() -> np.ndarray:
    """正交 DCT-II 基矩阵 A(8,8)，使 F = A @ X @ A.T。"""
    n = 8
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    A = np.cos((2 * j + 1) * i * np.pi / (2 * n))
    A[0] *= 1.0 / np.sqrt(2)
    return A * np.sqrt(2.0 / n)


_DCT_M = None
_IDCT_M = None


def _matrices():
    global _DCT_M, _IDCT_M
    if _DCT_M is None:
        a = _dct_matrix()
        _DCT_M, _IDCT_M = a, a.T
    return _DCT_M, _IDCT_M


def split_blocks(ch: np.ndarray) -> np.ndarray:
    """单通道 (H,W) float -> (H/8, W/8, 8, 8)。pad 到 8 的倍数。"""
    H, W = ch.shape
    Hp = int(np.ceil(H / BLOCK)) * BLOCK
    Wp = int(np.ceil(W / BLOCK)) * BLOCK
    p = np.zeros((Hp, Wp), dtype=ch.dtype)
    p[:H, :W] = ch
    # 边界用边缘复制填充，避免拉伸
    p[H:, :W] = ch[-1:, :]
    p[:H, W:] = ch[:, -1:]
    p[H:, W:] = ch[-1, -1]
    return p.reshape(Hp // BLOCK, Wp // BLOCK, BLOCK, BLOCK)


def join_blocks(blocks: np.ndarray, shape: tuple) -> np.ndarray:
    """blocks: (BH, BW, 8,8) -> (H,W) 裁剪回原尺寸。"""
    BH, BW, _, _ = blocks.shape
    img = blocks.transpose(0, 2, 1, 3).reshape(BH * BLOCK, BW * BLOCK)
    return img[: shape[0], : shape[1]]


def dct_blocks(blocks: np.ndarray) -> np.ndarray:
    """沿每块做 2D DCT。blocks:(...,8,8)。"""
    A, _ = _matrices()
    # 变形为 (N,8,8)
    s = blocks.shape
    x = blocks.reshape(-1, 8, 8)
    f = np.einsum("ij,njk,kl->nil", A, x, A, optimize=True)
    return f.reshape(s)


def idct_blocks(freq: np.ndarray) -> np.ndarray:
    """freq:(...,8,8) -> 空间块。"""
    _, AI = _matrices()
    s = freq.shape
    f = freq.reshape(-1, 8, 8)
    x = np.einsum("ij,njk,kl->nil", AI, f, AI, optimize=True)
    return x.reshape(s)


# ---- 标准量化表 (Annex K) ----
LUM_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)
CHROM_QT = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)


def scale_qtable(tab: np.ndarray, quality: int) -> np.ndarray:
    """libjpeg 风格质量缩放。quality 1..100。"""
    q = int(max(1, min(100, quality)))
    if q < 50:
        s = int(np.floor(5000 / max(1, q)))
    else:
        s = int(np.floor(200 - 2 * q))
    return np.clip(np.floor((tab * s + 50) / 100), 1, 255).astype(np.int16)


# ---- 之字顺序 ----
def _zigzag_idx() -> list:
    n = 8
    idx = [0] * (n * n)
    i = j = 0
    for k in range(n * n):
        idx[k] = i * n + j
        if (i + j) % 2 == 0:
            if j == n - 1:
                i += 1
            elif i == 0:
                j += 1
            else:
                i, j = i - 1, j + 1
        else:
            if i == n - 1:
                j += 1
            elif j == 0:
                i += 1
            else:
                i, j = i + 1, j - 1
    return idx


_ZIGZAG = None


def zigzag(blocks: np.ndarray) -> np.ndarray:
    """freq blocks:(...8,8) int -> 之字展平 (...,64)。"""
    global _ZIGZAG
    if _ZIGZAG is None:
        _ZIGZAG = np.array(_zigzag_idx())
    s = blocks.shape
    return blocks.reshape(-1, 64)[:, _ZIGZAG].reshape(s[:-2] + (64,))


def unzigzag(vals: np.ndarray) -> np.ndarray:
    """之字展平 (...,64) -> 块(...8,8)。"""
    global _ZIGZAG
    if _ZIGZAG is None:
        _ZIGZAG = np.array(_zigzag_idx())
    s = vals.shape
    out = vals.reshape((-1, 64))[..., np.argsort(_ZIGZAG)]
    return out.reshape(s[:-1] + (8, 8))