"""JPEG baseline 编解码器，封装为 YCC 对象（自包含 DCT+量化+Huffman）。

要点
----
- RGB -> YCbCr(4:2:0) -> 8x8 DCT -> 量化 -> 之字 -> 熵编码 -> 非交错多扫描 .jpg
- 构造时先偶数填充，使 chroma 子采样尺寸精确为 1/2，编码端块数与解码端由 header
  推算的块数一致 => "量化系数 -> 写文件 -> 读文件" 逐位相等（压缩域无失真往返）。
- 解码器解析 DQT/DHT/SOF0/SOS，支持通用 baseline(含非交错与交错)。
"""
from __future__ import annotations

import math
import struct

import numpy as np

from . import color as C
from . import dct as D
from . import huffman as H

SOI, EOI = 0xFFD8, 0xFFD9
SOF0, DHT, DQT, SOS, APP0 = 0xFFC0, 0xFFC4, 0xFFDB, 0xFFDA, 0xFFE0
COMP_SPEC = [(1, 0x22, 0), (2, 0x11, 1), (3, 0x11, 1)]  # Y,Cb,Cr


def _pad_even(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    ph, pw = h + (h & 1), w + (w & 1)
    if (ph, pw) == (h, w):
        return a
    out = np.zeros((ph, pw) + a.shape[2:], dtype=a.dtype)
    out[:h, :w] = a
    out[h:, :w] = a[-1:, :]
    out[:h, w:] = a[:, -1:]
    out[h:, w:] = a[-1, -1]
    return out


class YCC:
    def __init__(self, rgb: np.ndarray, quality: int = 85):
        a = _pad_even(np.asarray(rgb))
        self.orig_shape = (a.shape[0] - (a.shape[0] & 1), a.shape[1] - (a.shape[1] & 1))
        self.h, self.w = a.shape[:2]
        self.quality = quality
        ycc = C.rgb2ycbcr(a)
        self.qlum = D.scale_qtable(D.LUM_QT, quality)
        self.qchr = D.scale_qtable(D.CHROM_QT, quality)
        self.Y = self._forward(ycc[..., 0], self.qlum)
        self.Cb = self._forward(C.subsample(ycc[..., 1]), self.qchr)
        self.Cr = self._forward(C.subsample(ycc[..., 2]), self.qchr)

    def _forward(self, ch, qt):
        blocks = D.split_blocks(ch)
        return np.round(D.dct_blocks(blocks) / qt.astype(np.float64)[None, None]).astype(np.int16)

    # ----------------------------------------------------- 重建预览
    def reconstruct(self) -> np.ndarray:
        y = self._inv(self.Y, self.qlum)
        cb = C.upsample(self._inv(self.Cb, self.qchr)[: self.h // 2, : self.w // 2], (self.h, self.w))
        cr = C.upsample(self._inv(self.Cr, self.qchr)[: self.h // 2, : self.w // 2], (self.h, self.w))
        rgb = C.ycbcr2rgb(np.stack([y, cb, cr], axis=-1))
        return rgb[: self.orig_shape[0], : self.orig_shape[1]]

    def _inv(self, blocks, qt):
        freq = blocks.astype(np.float64) * qt.astype(np.float64)[None, None]
        img = D.join_blocks(D.idct_blocks(freq), (blocks.shape[0] * 8, blocks.shape[1] * 8))
        return np.clip(img + 128, 0, 255)

    # ----------------------------------------------------- 写 .jpg
    def to_bytes(self) -> bytes:
        out = bytearray()
        out += struct.pack(">H", SOI)
        app0 = bytearray(b"JFIF\x00\x01\x01\x00") + bytes(6)
        out += struct.pack(">H", APP0) + struct.pack(">H", len(app0) + 2) + app0
        dqt = H.build_quant_table_bytes(self.qlum, self.qchr, D.zigzag)
        out += struct.pack(">H", DQT) + struct.pack(">H", len(dqt) + 2) + dqt
        sof = bytearray([8]) + struct.pack(">HH", self.h, self.w) + bytes([3])
        for cid, hv, tq in COMP_SPEC:
            sof += bytes([cid, hv, tq])
        out += struct.pack(">H", SOF0) + struct.pack(">H", len(sof) + 2) + sof
        dht = H.dht_payload([(0, 0, H.DC_LUM), (1, 1, H.AC_LUM),
                             (0, 2, H.DC_CHR), (1, 3, H.AC_CHR)])
        out += struct.pack(">H", DHT) + struct.pack(">H", len(dht) + 2) + dht
        for arr, cid, dc, ac in ((self.Y, 1, H.DC_LUM, H.AC_LUM),
                                 (self.Cb, 2, H.DC_CHR, H.AC_CHR),
                                 (self.Cr, 3, H.DC_CHR, H.AC_CHR)):
            # 非交错扫描 SOS：Ns + 分量(cid, td<<4|ta) + Ss + Se + AhAl
            td = 0 if cid == 1 else 2
            ta = 1 if cid == 1 else 3
            sos = bytes([1, cid, (td << 4) | ta, 0, 63, 0])
            out += struct.pack(">H", SOS) + struct.pack(">H", len(sos) + 2) + sos
            out += _encode_blocks(arr, dc, ac)
        out += struct.pack(">H", EOI)
        return bytes(out)

    # ----------------------------------------------------- 读 .jpg
    @classmethod
    def from_bytes(cls, data: bytes):
        frame, qt, dht, (h, w) = _scan_markers(data)
        grid = _decode_entropy(data, frame, dht, (h, w))
        obj = cls.__new__(cls)
        obj.h, obj.w, obj.orig_shape, obj.quality = h, w, (h, w), 85
        obj.qlum, obj.qchr = qt.get(0, np.zeros((8, 8), np.int16)), qt.get(1, np.zeros((8, 8), np.int16))
        obj.Y, obj.Cb, obj.Cr = grid[1], grid[2], grid[3]
        return obj


# ------------------------------------------------------------- 熵编码
def _encode_blocks(qblocks: np.ndarray, dc_tab=None, ac_tab=None) -> bytes:
    bw = H.BitWriter()
    dc_tab = dc_tab or H.DC_TABLE
    ac_tab = ac_tab or H.AC_TABLE
    blocks = D.zigzag(qblocks.astype(np.int64)).reshape(-1, 64)
    prev = 0
    for z in blocks:
        diff = int(z[0]) - prev
        prev = int(z[0])
        if diff:
            size, mag = _cat_mag(diff)
            c, l = dc_tab.code_of(size)
            bw.put(c, l); bw.put(mag & ((1 << size) - 1), size)
        else:
            c, l = dc_tab.code_of(0)
            bw.put(c, l)
        run = 0
        last = 0
        for i in range(1, 64):
            v = int(z[i])
            if v == 0:
                run += 1
                continue
            while run >= 16:
                c, l = ac_tab.code_of(0xF0)
                bw.put(c, l); run -= 16
            s, mag = _cat_mag(v)
            c, l = ac_tab.code_of((run << 4) | s)
            bw.put(c, l); bw.put(mag & ((1 << s) - 1), s)
            run = 0
            last = i
        # 仅当存在尾部零系数时才写 EOB；满块(k==64)时解码端不会再读 EOB
        if last < 63:
            c, l = ac_tab.code_of(0)
            bw.put(c, l)
    bw.pad_final()
    return bw.buf


def _cat_mag(v: int):
    av = abs(v)
    size = 0
    while (1 << size) - 1 < av:
        size += 1
    return size, (v - 1 if v < 0 else v)


# ------------------------------------------------------------- 解码
def _scan_markers(data: bytes):
    """扫描 header 段；返回 (frame, qt, dht, shape)。"""
    frame, qt, dht, shape = [], {}, {}, (0, 0)
    pos = 2
    while pos < len(data):
        while data[pos] != 0xFF:
            pos += 1
        marker = struct.unpack(">H", data[pos:pos + 2])[0]
        if marker == EOI or marker == SOS:
            break
        if marker == SOF0:
            ln = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            p = data[pos + 4:pos + 2 + ln]
            shape = struct.unpack(">HH", p[1:5])
            i = 6
            while i + 2 < len(p):
                frame.append({"id": p[i], "h": p[i + 1] >> 4, "v": p[i + 1] & 0xF, "tq": p[i + 2]})
                i += 3
            pos += 2 + ln
        elif marker == DQT:
            ln = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            p = data[pos + 4:pos + 2 + ln]
            i = 0
            while i < len(p):
                qid = p[i] & 0x0F
                vals = np.frombuffer(p[i + 1:i + 65], dtype=np.uint8).astype(np.int16)
                i += 65
                qt[qid] = D.unzigzag(vals)
            pos += 2 + ln
        elif marker == DHT:
            ln = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            p = data[pos + 4:pos + 2 + ln]
            i = 0
            while i < len(p):
                klass, tid = p[i] >> 4, p[i] & 0x0F
                bits = list(p[i + 1:i + 17]); i += 17
                n = sum(bits)
                hv = list(p[i:i + n]); i += n
                dht[(klass, tid)] = H.HuffTable(bits, hv)
            pos += 2 + ln
        else:
            ln = struct.unpack(">H", data[pos + 2:pos + 4])[0]
            pos += 2 + ln
    return frame, qt, dht, shape


def _decode_entropy(data: bytes, frame, dht, frame_shape):
    cmap = {c["id"]: c for c in frame}
    max_h = max((c["h"] for c in frame), default=1)
    max_v = max((c["v"] for c in frame), default=1)
    hh, ww = frame_shape
    p = 2
    out = {}
    while p < len(data):
        while data[p] != 0xFF:
            p += 1
        marker = struct.unpack(">H", data[p:p + 2])[0]
        if marker == EOI:
            break
        if marker == SOS:
            ln = struct.unpack(">H", data[p + 2:p + 4])[0]
            head = data[p + 4:p + 2 + ln]
            nc = head[0]
            sel = []
            i = 1
            for _ in range(nc):
                cid, tdta = head[i], head[i + 1]
                sel.append({"id": cid, "td": tdta >> 4, "ta": tdta & 0x0F})
                i += 2
            start = p + 2 + ln
            end = start
            while end < len(data) - 1 and not (data[end] == 0xFF and data[end + 1] != 0x00):
                end += 1
            ent = data[start:end]
            _fill(out, ent, sel, cmap, dht, hh, ww, max_h, max_v)
            p = end
        else:
            ln = struct.unpack(">H", data[p + 2:p + 4])[0]
            p += 2 + ln
    for cid, ci in cmap.items():
        if cid not in out:
            bx, by = comp_blocks(ci, hh, ww, max_h, max_v)
            out[cid] = np.zeros((by, bx, 8, 8), dtype=np.int16)
    return out


def comp_blocks(ci, hh, ww, max_h, max_v):
    px = int(math.ceil(ww * ci["h"] / max_h))
    py = int(math.ceil(hh * ci["v"] / max_v))
    return int(math.ceil(px / 8)), int(math.ceil(py / 8))


def _fill(out, ent, sel, cmap, dht, hh, ww, max_h, max_v):
    br = H.BitReader(ent)
    if len(sel) == 1:
        cid = sel[0]["id"]
        ci = cmap[cid]
        t0 = dht.get((0, sel[0]["td"]), H.DC_TABLE)
        t1 = dht.get((1, sel[0]["ta"]), H.AC_TABLE)
        bx, by = comp_blocks(ci, hh, ww, max_h, max_v)
        arr = _decode_n(br, bx * by, t0, t1)      # (nb,64) 之字序
        out[cid] = D.unzigzag(arr).reshape(by, bx, 8, 8).copy()
    else:
        # 交错 baseline: 逐 MCU, 块序先竖直后水平
        # 分量在 MCU 内的实际块尺寸 = 其在该分量的网格, 这里按 HxV 摆放
        tmp = {s["id"]: [] for s in sel}
        mcu_x = _mcu_count(ww, max_h)
        mcu_y = _mcu_count(hh, max_v)
        for _ in range(mcu_x * mcu_y):
            for s in sel:
                ci = cmap[s["id"]]
                t0 = dht.get((0, s["td"]), H.DC_TABLE)
                t1 = dht.get((1, s["ta"]), H.AC_TABLE)
                tmp[s["id"]].append(_decode_one(br, t0, t1))
        for cid, bl in tmp.items():
            ci = cmap[cid]
            by = mcu_y * ci["v"]
            bx = mcu_x * ci["h"]
            grid = np.zeros((by, bx, 8, 8), dtype=np.int16)
            k = 0
            for my in range(mcu_y):
                for mx in range(mcu_x):
                    for vy in range(ci["v"]):
                        for hx in range(ci["h"]):
                            grid[my * ci["v"] + vy, mx * ci["h"] + hx] = D.unzigzag(bl[k][None])[0]
                            k += 1
            out[cid] = grid


def _mcu_count(dim, factor):
    return int(math.ceil(dim / (factor * 8)))


def _decode_one(br, t0, t1):
    z = np.zeros(64, dtype=np.int16)
    size = t0.decode(br)
    if size:
        z[0] = H._dec_value(br.get(size), size)
    else:
        z[0] = 0
    k = 1
    while k < 64:
        rs = t1.decode(br)
        r, s = (rs >> 4) & 0x0F, rs & 0x0F
        if s == 0:
            if r == 0:
                break
            if r == 15:
                k += 16
                continue
            break
        k += r
        if k < 64:
            z[k] = H._dec_value(br.get(s), s)
            k += 1
    return z


def _decode_n(br, nb, t0, t1):
    prev = 0
    out = np.zeros((nb, 64), dtype=np.int16)
    for i in range(nb):
        z = out[i]
        size = t0.decode(br)
        if size:
            prev += H._dec_value(br.get(size), size)
        z[0] = prev
        k = 1
        while k < 64:
            rs = t1.decode(br)
            r, s = (rs >> 4) & 0x0F, rs & 0x0F
            if s == 0:
                if r == 0:
                    break
                if r == 15:
                    k += 16
                    continue
                break
            k += r
            if k < 64:
                z[k] = H._dec_value(br.get(s), s)
                k += 1
    return out