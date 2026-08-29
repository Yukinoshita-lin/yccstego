"""面向文件/数组的高层 API：嵌入、提取、分析。

流程：RGB → YCC（DCT+量化+Huffman 压缩到 .jpg 位流）→ 在量化 Y 的
非零 AC 系数上做 nsF5 伴随式嵌入 → 写回位流；提取端解析位流回到量化系数再取消息。
由于压缩域往返是"逐位一致"的（见 jpeg_codec），嵌入系数在文件往返后保持不变。
"""
from __future__ import annotations

import numpy as np

from . import jpeg_codec as J
from . import nsf5
from . import steganalysis as S


def _as_rgb(image) -> np.ndarray:
    if isinstance(image, np.ndarray):
        a = image
    elif isinstance(image, (bytes, bytearray)):
        from PIL import Image
        import io
        a = np.asarray(Image.open(io.BytesIO(bytes(image))).convert("RGB"))
    else:  # 假定是路径 / 兼容 Pillow 对象
        from PIL import Image
        a = np.asarray(Image.open(image).convert("RGB"))
    return np.ascontiguousarray(a.astype(np.uint8))


def embed_bytes(rgb_image, text: str, p: int = 3, password: str = "", quality: int = 85,
                truncate: bool = False):
    """在图像上嵌入任意文本（UTF-8，中英文均可），返回 (.jpg bytes, report)。

    rgb_image 支持 ndarray / bytes / 路径 / Pillow 对象。返回的字节是带隐写载荷的
    标准 JPEG（libjpeg/Pillow 可打开），从位流本身也可无损还原嵌入系数。
    消息超出容量时：truncate=False 抛 CapacityError；truncate=True 安全截断。
    """
    rgb = _as_rgb(rgb_image)
    c0 = J.YCC(rgb, quality)
    y_new, rep = nsf5.embed_into_y(c0.Y, c0.Cb, c0.Cr, text, p=p, password=password,
                                   truncate=truncate)
    c1 = J.YCC.__new__(J.YCC)
    c1.h, c1.w, c1.orig_shape, c1.quality = c0.h, c0.w, c0.orig_shape, quality
    c1.qlum, c1.qchr = c0.qlum, c0.qchr
    c1.Y, c1.Cb, c1.Cr = y_new, c0.Cb, c0.Cr
    return c1.to_bytes(), rep


def extract_bytes(jpg_bytes, p: int = 3, password: str = ""):
    """从 .jpg 位流提取消息。返回 (message|None, cover_hash_hex, tampered, head_match)。"""
    y = J.YCC.from_bytes(bytes(jpg_bytes))
    return nsf5.extract_from_y(y.Y, y.Cb, y.Cr, p=p, password=password)


def analyze_bytes(jpg_bytes, sensitivity="均衡", unit_baseline=None):
    """对 .jpg 位流做盲隐写分析。返回像素域与 DCT 域两套结论的合并字典。"""
    y = J.YCC.from_bytes(bytes(jpg_bytes))
    dct = S.analyze_y(y.Y, sensitivity=sensitivity, unit_baseline=unit_baseline)
    try:
        pix = S.analyze(y.reconstruct(), sensitivity=sensitivity)
    except Exception:
        pix = None
    out = dict(dct)
    if pix:
        out.update(pix_probability=pix["stego_probability"],
                   pix_verdict=pix["verdict"], pix=pix)
    return out