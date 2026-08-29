"""JPEG-DCT 编解码(BT.601 YCbCr / 量化 / 之字 / Huffman 熵编码)单元测试。

覆盖:
- YCC 全流程往返: 量化系数逐位一致
- 熵编码跨实现兼容: 自产 .jpg 能被 Pillow/libjpeg 解码
- ZRL(≥16连续零)场景
- DC 类别的合法边界(标准 DC 亮度表最大类别 11, |DC|<=2047)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yccstego import jpeg_codec as J
from yccstego import color, dct, huffman as H
from yccstego.jpeg_codec import _encode_blocks, _decode_n, _cat_mag
from yccstego.huffman import BitReader


def make_rgb(h, w, seed=1, rng_np=np.random.RandomState):
    rng = rng_np(seed)
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    base = np.sin(3 * x) * np.cos(5 * y) * 60 + 120
    R = np.clip(base + 3 * x * 255 + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    G = np.clip(120 + 60 * np.sin(8 * x) * np.exp(-(y - 0.5) ** 2 * 20)
                + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    B = np.clip(base - 40 * x + rng.normal(0, 8, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([R, G, B], axis=-1)


def _entropy_roundtrip(blocks):
    """编码->解码后量化块逐位一致。blocks:(nb,8,8) int16。"""
    nb = blocks.shape[0]
    data = _encode_blocks(blocks)
    back = _decode_n(BitReader(data), nb, H.DC_TABLE, H.AC_TABLE)
    expect = dct.zigzag(blocks).reshape(nb, 64)
    return np.array_equal(expect, back), expect, back


class TestJPEGRoundtrip(unittest.TestCase):
    def test_ycc_roundtrip_bitwise(self):
        for q in (70, 85, 95):
            rgb = make_rgb(96, 128, seed=q)
            ycc = J.YCC(rgb, quality=q)
            back = J.YCC.from_bytes(ycc.to_bytes())
            self.assertTrue(np.array_equal(ycc.Y, back.Y), f"Y mismatch q={q}")
            self.assertTrue(np.array_equal(ycc.Cb, back.Cb), f"Cb mismatch q={q}")
            self.assertTrue(np.array_equal(ycc.Cr, back.Cr), f"Cr mismatch q={q}")

    def test_pillow_decodes_our_jpg(self):
        from PIL import Image
        import tempfile
        rgb = make_rgb(48, 64, seed=5)
        ycc = J.YCC(rgb, quality=85)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(ycc.to_bytes())
            path = f.name
        try:
            im = Image.open(path)
            im.load()
            self.assertEqual(im.size, (64, 48))
        finally:
            os.unlink(path)

    def test_reconstruct_shape(self):
        rgb = make_rgb(80, 96, seed=7)
        ycc = J.YCC(rgb, quality=85)
        pre = ycc.reconstruct()
        self.assertEqual(pre.shape, rgb.shape)

    def test_quality_scales_capacity(self):
        rgb = make_rgb(64, 64, seed=9)
        nz70 = np.count_nonzero(J.YCC(rgb, 70).Y)
        nz95 = np.count_nonzero(J.YCC(rgb, 95).Y)
        self.assertGreater(nz95, nz70)


class TestEntropyCoding(unittest.TestCase):
    def test_random_dc_valid_range(self):
        # 差分编码: 每步 DC 差值须落在标准 DC 亮度表类别范围内(|diff|<=2047 => 类别<=11)。
        # 相邻块的 DC 差即 Huffman 差分编码量, 直接以合法差值随机游走生成。
        rng = np.random.RandomState(0)
        for trial in range(5):
            nb = 3 * (trial + 1)
            blk = np.zeros((nb, 8, 8), dtype=np.int16)
            prev = 0
            for i in range(nb):
                step = int(rng.randint(-1500, 1501))
                dc = prev + step
                dc = int(np.clip(dc, -2047, 2047))
                blk[i, 0, 0] = dc
                prev = dc
                for _ in range(rng.randint(0, 8)):
                    k = rng.randint(1, 64)
                    blk[i, k // 8, k % 8] = rng.choice([-7, -5, -3, -1, 1, 2, 3, 5])
            ok, _, _ = _entropy_roundtrip(blk)
            self.assertTrue(ok, f"entropy roundtrip mismatch trial {trial}")

    def test_zrl_long_runs(self):
        for zeros in (16, 17, 20, 31, 40):
            blk = np.zeros((1, 8, 8), dtype=np.int16)
            blk[0, 0, 0] = 100
            idx = 1 + zeros
            blk[0, idx // 8, idx % 8] = 3
            ok, _, _ = _entropy_roundtrip(blk)
            self.assertTrue(ok, f"ZRL={zeros} roundtrip mismatch")

    def test_cat_mag_boundaries(self):
        # DC 类别边界: 1024..2047 → 类别11; 2048 不合法(无类别12)
        self.assertEqual(_cat_mag(1024)[0], 11)
        self.assertEqual(_cat_mag(2047)[0], 11)
        self.assertEqual(_cat_mag(1)[0], 1)
        self.assertEqual(_cat_mag(0)[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)