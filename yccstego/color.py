"""RGB <-> YCbCr(BT.601) 转换与 4:2:0 子采样。"""
from __future__ import annotations

import numpy as np

# BT.601 full-swing 系数 (0..255)
KR, KG, KB = 0.114, 0.587, 0.299


def rgb2ycbcr(rgb: np.ndarray) -> np.ndarray:
    """rgb: (H,W,3) uint8 -> ycbr: (H,W,3) float32 [0,255]。"""
    R = rgb[..., 0].astype(np.float32)
    G = rgb[..., 1].astype(np.float32)
    B = rgb[..., 2].astype(np.float32)
    Y = KR * R + KG * G + KB * B
    Cb = 128.0 + (B - Y) * 0.564
    Cr = 128.0 + (R - Y) * 0.713
    return np.stack([Y, Cb, Cr], axis=-1)


def ycbcr2rgb(ycc: np.ndarray) -> np.ndarray:
    """ycc: (H,W,3) float -> rgb uint8 Clamped。"""
    Y = ycc[..., 0].astype(np.float32)
    Cb = ycc[..., 1].astype(np.float32) - 128.0
    Cr = ycc[..., 2].astype(np.float32) - 128.0
    R = Y + 1.402 * Cr
    G = Y - 0.344136 * Cb - 0.714136 * Cr
    B = Y + 1.772 * Cb
    out = np.stack([R, G, B], axis=-1)
    return np.clip(out, 0, 255).round().astype(np.uint8)


def subsample(ch: np.ndarray, factor: int = 2) -> np.ndarray:
    """4:2:0: 对单通道做 2x2 均值下采样。ch: (H,W) float。"""
    H, W = ch.shape
    Hs, Ws = H // factor, W // factor
    ch = ch[: Hs * factor, : Ws * factor]
    return 0.25 * (ch[::2, ::2] + ch[1::2, ::2] + ch[::2, 1::2] + ch[1::2, 1::2])


def upsample(ch: np.ndarray, target: tuple, factor: int = 2) -> np.ndarray:
    """最近邻放大回原尺寸。chn: (Hs,Ws)。"""
    H, W = target
    return np.repeat(np.repeat(ch, factor, axis=0), factor, axis=1)[:H, :W]