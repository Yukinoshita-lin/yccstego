"""JPEG 熵编码/解码：BitWriter/BitReader 与哈夫曼表（双端对称固定位宽，完全可逆）。

为规避手抄标准 162 符号 AC 表的转写风险，这里在合法的 JPEG Huffman 框架下使用
**固定位宽且完备**的码表：DC 用 4bit(16 符号)、AC 用 8bit(256 符号)。二者都是满足
Kraft 等式的完整二叉树，external 解码器(Pillow/libjpeg) 依据 DHT 表即可正确解码；
编码/解码由同一表完成，保证 JPEG 往返逐位一致。
"""
from __future__ import annotations

import numpy as np

# 固定位宽：DC len=4 (16 符号), AC len=8 (256 符号)
DC_LEN = 4
AC_LEN = 8


class BitWriter:
    """MSB-first bit writer。写完成熵数据后用 0xFF00 填充转义。"""

    def __init__(self):
        self.buf = bytearray()
        self.acc = 0x00
        self.nbits = 0

    def put(self, value: int, nbits: int) -> None:
        for i in range(nbits - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.nbits += 1
            if self.nbits == 8:
                self.buf.append(self.acc)
                self.acc, self.nbits = 0, 0

    def align_byte(self) -> None:
        while self.nbits:
            self.acc = (self.acc << 1) | 1
            self.nbits += 1
            if self.nbits == 8:
                self.buf.append(self.acc)
                self.acc, self.nbits = 0, 0

    def pad_final(self) -> None:
        self.align_byte()
        # 0xFF 后补 0x00 转义，保证不与 marker 冲突
        out = bytearray()
        for b in self.buf:
            out.append(b)
            if b == 0xFF:
                out.append(0x00)
        self.buf = out


class BitReader:
    def __init__(self, data: bytes):
        self.buf = data
        self.pos = 0
        self.byte = 0
        self.nbits = 0

    def get(self, nbits: int) -> int:
        v = 0
        for _ in range(nbits):
            v = (v << 1) | self.get1()
        return v

    def get1(self) -> int:
        if self.nbits == 0:
            # 跳过 0xFF00 转义：FF 后继是 00（非 marker）则视为数据 0xFF
            b = self.buf[self.pos]
            self.pos += 1
            if b == 0xFF:
                b2 = self.buf[self.pos]
                self.pos += 1
                b = 0xFF  # luma value; 0x00 是正文
            self.byte = b
            self.nbits = 8
        self.nbits -= 1
        return (self.byte >> self.nbits) & 1


class HuffTable:
    """由 bits[16] + huffval 构建的哈夫曼表；支持编码与解码。"""

    def __init__(self, bits: list, huffval: list):
        self.huffval = huffval
        self.sym_to_code = {}
        self.maxcode = {}
        self.mincode = {}
        self.valptr = {}
        code = 0
        k = 0
        for l in range(1, 17):
            n = bits[l - 1] if l <= 16 else 0
            if n:
                self.mincode[l] = code
                self.valptr[l] = k
            else:
                self.mincode[l] = code
                self.maxcode[l] = -1
            for _ in range(n):
                self.sym_to_code[huffval[k]] = (code, l)
                code += 1
                k += 1
            if n:
                self.maxcode[l] = code - 1
            # 空码长层同样左移：与 libjpeg 规范一致，避免 15/16 层码与短码前缀冲突。
            code <<= 1

    def code_of(self, sym: int):
        return self.sym_to_code[sym]

    def decode(self, br: BitReader) -> int:
        code = 0
        for l in range(1, 17):
            code = (code << 1) | br.get1()
            if code <= self.maxcode.get(l, -1):
                return self.huffval[self.valptr[l] + code - self.mincode[l]]
        raise ValueError("invalid Huffman code")

    @staticmethod
    def uniform(nsym: int, bitlen: int):
        """固定位宽完备表。"""
        n = 1 << bitlen
        bits = [0] * 16
        bits[bitlen - 1] = n
        return HuffTable(bits, list(range(n)))


# ---- 标准 JPEG Huffman 表 (Annex K.3.3, 与 libjpeg 完全一致) ----
# 直流表: 分类符号 0..11; 交流表: 162 个 (run<<4)|size 符号。可直接被 libjpeg 解码。
_DC_LUM_BITS  = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
_DC_LUM_VAL   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_DC_CHR_BITS  = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
_DC_CHR_VAL   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
_AC_LUM_BITS  = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125]
_AC_LUM_VAL   = [1, 2, 3, 0, 4, 17, 5, 18, 33, 49, 65, 6, 19, 81, 97, 7, 34, 113,
                 20, 50, 129, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240,
                 36, 51, 98, 114, 130, 9, 10, 22, 23, 24, 25, 26, 37, 38, 39, 40,
                 41, 42, 52, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71, 72, 73,
                 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103, 104,
                 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133,
                 134, 135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153,
                 154, 162, 163, 164, 165, 166, 167, 168, 169, 170, 178, 179, 180,
                 181, 182, 183, 184, 185, 186, 194, 195, 196, 197, 198, 199, 200,
                 201, 202, 210, 211, 212, 213, 214, 215, 216, 217, 218, 225, 226,
                 227, 228, 229, 230, 231, 232, 233, 234, 241, 242, 243, 244, 245,
                 246, 247, 248, 249, 250]
_AC_CHR_BITS  = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119]
_AC_CHR_VAL   = [0, 1, 2, 3, 17, 4, 5, 33, 49, 6, 18, 65, 81, 7, 97, 113,
                 19, 34, 50, 129, 8, 20, 66, 145, 161, 177, 193, 9, 35, 51, 82,
                 240, 21, 98, 114, 209, 10, 22, 36, 52, 225, 37, 241, 23, 24,
                 25, 26, 38, 39, 40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69,
                 70, 71, 72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100,
                 101, 102, 103, 104, 105, 106, 115, 116, 117, 118, 119, 120, 121,
                 122, 130, 131, 132, 133, 134, 135, 136, 137, 138, 146, 147, 148,
                 149, 150, 151, 152, 153, 154, 162, 163, 164, 165, 166, 167, 168,
                 169, 170, 178, 179, 180, 181, 182, 183, 184, 185, 186, 194, 195,
                 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214, 215,
                 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234, 242,
                 243, 244, 245, 246, 247, 248, 249, 250]

DC_LUM = HuffTable(_DC_LUM_BITS, _DC_LUM_VAL)
AC_LUM = HuffTable(_AC_LUM_BITS, _AC_LUM_VAL)
DC_CHR = HuffTable(_DC_CHR_BITS, _DC_CHR_VAL)
AC_CHR = HuffTable(_AC_CHR_BITS, _AC_CHR_VAL)
# 兼容旧引用的默认表(亮度表)
DC_TABLE = DC_LUM
AC_TABLE = AC_LUM


def _cat_mag(v: int):
    """返回 (size, 幅度位)。v!=0。编码用 v 或 v-1。"""
    av = abs(v)
    size = 0
    while (1 << size) - 1 < av:
        size += 1
    mag = v - 1 if v < 0 else v
    return size, mag


def _enc_mag(mag: int, size: int) -> int:
    """把可能为负的 mag 规约到 size 位无符号。"""
    return mag & ((1 << size) - 1)


def _dec_value(val: int, size: int) -> int:
    """按符号位还原系数：最高位=1 为正，=0 为负。"""
    if size == 0:
        return 0
    if val >= (1 << (size - 1)):
        return val
    return val - (1 << size) + 1


def write_segment(buf, marker: int, payload: bytes):
    buf.extend(marker.to_bytes(2, "big"))
    buf.extend((len(payload) + 2).to_bytes(2, "big"))
    buf.extend(payload)


def dqt_payload(qt: np.ndarray, tq: int, zz: np.ndarray) -> bytes:
    p = bytearray([tq])
    p.extend(zz(qt.astype(np.int16).reshape(8, 8).flat).astype(np.uint8).tobytes())
    return bytes(p)


def build_quant_table_bytes(qt_lum, qt_chrom, zz) -> bytes:
    p = bytearray()
    p.append(0)  # table id 0 (luma)
    p.extend(zz(qt_lum.astype(np.int16).reshape(8, 8)).astype(np.uint8).tobytes())
    p.append(1)  # table id 1 (chroma)
    p.extend(zz(qt_chrom.astype(np.int16).reshape(8, 8)).astype(np.uint8).tobytes())
    return bytes(p)


def dht_payload(tables: list) -> bytes:
    """tables: list of (klass:int, id:int, HuffTable)。"""
    p = bytearray()
    for klass, tid, tb in tables:
        p.append((klass << 4) | tid)
        bits = [0] * 16
        for sym, (_, l) in tb.sym_to_code.items():
            bits[l - 1] += 1
        p.extend(bits)
        p.extend(tb.huffval)
    return bytes(p)