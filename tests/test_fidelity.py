"""图像保真度回归 (2026-10-04 制作宣传片素材时发现)。

历史 bug 组合会产出"品红条纹" JPEG —— 消息能提取, 但肉眼看到的图是坏的:
1. color.KR/KB 写反 (BT.601 红/蓝权重互换);
2. YCC._forward 缺 JPEG 标准 -128 电平偏移, _inv 却加回;
3. dct.split_blocks 直接 reshape, 未做块转置, 8x8 块内容被打乱;
4. dct_blocks/idct_blocks 用 A·x·A (自洽但非标准 2D DCT-II)。
本文件锁定四类回归: 颜色恒等 / 块布局互逆 / 标准 DCT 形状 / 全链路 PSNR。
"""
import io
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yccstego import color, dct, jpeg_codec as J  # noqa: E402


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def make_rgb(h, w):
    """确定性彩色渐变 (覆盖全亮度区间与强色度)。"""
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    R = (255 * x).astype(np.uint8)
    G = (255 * y).astype(np.uint8)
    B = (255 * np.sqrt(np.clip(1 - x * y, 0, 1))).astype(np.uint8)
    return np.stack([np.broadcast_to(R, (h, w)),
                     np.broadcast_to(G, (h, w)),
                     np.broadcast_to(B, (h, w))], axis=-1)


class TestColor(unittest.TestCase):
    def test_known_colors(self):
        px = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8)
        ycc = color.rgb2ycbcr(px)
        self.assertAlmostEqual(float(ycc[0, 0, 0]), 76.2, places=1)  # 红 -> Y
        np.testing.assert_array_equal(color.ycbcr2rgb(ycc), px)

    def test_gray_stays_gray(self):
        ycc = color.rgb2ycbcr(np.full((16, 16, 3), 137, np.uint8))
        self.assertAlmostEqual(float(ycc[..., 1].mean()), 128.0, places=3)
        self.assertAlmostEqual(float(ycc[..., 2].mean()), 128.0, places=3)


class TestBlocks(unittest.TestCase):
    def test_split_join_inverse(self):
        ch = np.random.default_rng(3).normal(128, 40, (67, 45))  # 非 8 倍数, 触发 pad
        blocks = dct.split_blocks(ch)
        back = dct.join_blocks(blocks, (72, 48))
        self.assertTrue(np.array_equal(back[:67, :45], ch))      # 原区域无损
        # 每个块 == 原平面 (含边缘复制 pad) 的对应 8x8 窗口
        p = np.zeros((72, 48))
        p[:67, :45] = ch
        p[67:, :45] = ch[-1:, :]
        p[:67, 45:] = ch[:, -1:]
        p[67:, 45:] = ch[-1, -1]
        for i in range(9):
            for j in range(6):
                self.assertTrue(np.array_equal(blocks[i, j], p[i*8:i*8+8, j*8:j*8+8]))

    def test_split_layout(self):
        ch = np.arange(384, dtype=np.float64).reshape(16, 24)
        self.assertEqual(int(dct.split_blocks(ch)[0, 1, 0, 2]), int(ch[0, 10]))

    def test_dct_is_standard(self):
        c = np.full((2, 8, 8), 100.0)      # 常数块 -> 仅 DC 非零, DC = 8 * 常数
        f = dct.dct_blocks(c)
        self.assertEqual(int((np.abs(f) > 1e-8).sum()), 2)
        self.assertAlmostEqual(float(f[0, 0, 0]), 800.0, places=6)

    def test_dct_idct_identity(self):
        x = np.random.default_rng(4).normal(size=(5, 8, 8))
        self.assertTrue(np.allclose(dct.idct_blocks(dct.dct_blocks(x)), x, atol=1e-9))


class TestEndToEndFidelity(unittest.TestCase):
    def test_codec_reconstruct_psnr(self):
        rgb = make_rgb(96, 96)
        r = J.YCC(rgb, 85).reconstruct()
        self.assertGreater(psnr(rgb, r), 30.0)

    def test_libjpeg_pixel_fidelity(self):
        """自产 .jpg 不止"能打开", 解出的像素也要对。"""
        rgb = make_rgb(96, 96)
        jpg = J.YCC(rgb, 85).to_bytes()
        back = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB"))
        self.assertGreater(psnr(rgb, back), 30.0)


if __name__ == "__main__":
    unittest.main()
