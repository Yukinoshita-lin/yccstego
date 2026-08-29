"""Y 亮度量化 DCT 系数上的 nsF5 隐写核心（伴随式矩阵编码 + 湿纸 + 图像哈希自同步）。

载体系数是**量化 Y 块的非零 AC 系数**（块内直流不参与），载体比特 = 系数整数 LSB 奇偶
x = c & 1。F5 语义是"幅值减 1（保符号）"翻 LSB；但 |c|==1 时减幅会归零（收缩），而
收缩会改变载体系数集合、破坏嵌入/提取的对齐 —— 这正是 F5 的"收缩"问题。

因此这里用 **nsF5 湿纸编码**：把 |c|==1 的系数视为"湿"位，绝不触碰；只在"干"位
（|c|>1）上解 GF(2) 伴随式方程，用权重尽量低的解完成编码 → 无收缩，载体系数集合在
嵌入前后完全相同，可逐位可逆还原。极端无解时兜底把目标系数幅值升到 2（仍翻 LSB、
保持非零不进 0），保证必然可解码。

图像哈希自同步 / 篡改感知
-------------------------
对**不参与嵌入的 Cb+Cr 量化系数**计算 SHA-256（cover_hash，截 16 字节）：
- 头部池：载体系数序最前 N_h 个，用仅"口令"派生的 seed0 置乱，预埋 cover_hash；
- 正文池：其余载体系数，用 cover_hash+口令 派生的 seed 键控置乱。
提取端先读头部取回 cover_hash，再重算正文路径。由于 cover_hash 独立刻画图像内容：
任改一个色度系数 → 重算哈希与取回值失配（篡改感知）；任改正文区 Y 系数 → 载体列表
或长度头 / ASCII 校验失败。头部池与正文池互不相交，正文嵌入不会覆盖头部。
"""

from __future__ import annotations

import hashlib
from typing import Optional, Sequence

import numpy as np

MSG_HEADER_BITS = 16    # UTF-8 字节数（两字节长度头，≤65535；中文每字 3 字节）
COVER_HASH_BYTES = 16   # 认证头：sha256 截取前 16 字节（128 bit）


class CapacityError(RuntimeError):
    """载体系数不足，无法容纳所要求长度的消息。"""


# --------------------------------------------------------------------------- #
#  二元汉明码 [n,p], n = 2^p - 1，H：(p,n)，列为 GF(2)^p 全部非零向量
# --------------------------------------------------------------------------- #
def build_hamming(p: int) -> np.ndarray:
    if p < 1:
        raise ValueError("p 必须 >= 1")
    n = (1 << p) - 1
    H = np.zeros((p, n), dtype=np.uint8)
    for j in range(1, n + 1):
        for r in range(p):
            H[r, j - 1] = (j >> r) & 1
    return H


def syndrome(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """s = H·x (mod 2)。x:(n,) → (p,)。"""
    return (H @ (x & 1)) & 1


# --------------------------------------------------------------------------- #
#  种子 / 置换（纯 Python splitmix64 + Fisher-Yates，确定性、嵌入/提取一致）
# --------------------------------------------------------------------------- #
MASK64 = (1 << 64) - 1


def derive_seed(image_bytes: bytes, password: str = "") -> int:
    base = hashlib.sha256(image_bytes).hexdigest()
    if password:
        base = hashlib.sha256(base.encode() + password.encode()).hexdigest()
    return int(base, 16)


def _splitmix(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def permute_index(total: int, seed: int) -> np.ndarray:
    """确定性伪随机置换 [0..total)。算法固定，改动会破坏嵌入/提取可逆性。"""
    a = np.arange(total, dtype=np.int64)
    state = int(seed) & MASK64
    for i in range(total - 1, 0, -1):
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        r = _splitmix(state)
        j = int(r % (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


# --------------------------------------------------------------------------- #
#  消息编码：字符串 ↔ 比特（UTF-8，16 位字节长度头）
# --------------------------------------------------------------------------- #
def encode_string(text: str) -> np.ndarray:
    raw = text.encode("utf-8")   # 任意 Unicode（中文等）→ UTF-8 字节流
    length = len(raw)
    if length > 0xFFFF:
        raise ValueError("文本过长（≤65535 字节）")
    head = np.array([(length >> i) & 1 for i in range(MSG_HEADER_BITS)], np.uint8)
    body = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return np.concatenate([head, body])


def decode_string(bits: np.ndarray) -> str:
    if bits.size < MSG_HEADER_BITS:
        raise ValueError("数据过短，无法解码长度头")
    length = sum(int(bits[i]) << i for i in range(MSG_HEADER_BITS))
    body = bits[MSG_HEADER_BITS:MSG_HEADER_BITS + length * 8]
    if body.size < length * 8:
        raise ValueError("有效载荷不足，消息可能被截断或被破坏")
    try:
        return bytes(np.packbits(body[:length * 8])).decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("提取字节非法（UTF-8），消息可能被篡改或已损坏")


# --------------------------------------------------------------------------- #
#  湿纸求解：在"干"列上找尽量稀疏的解 H[:,dry]·y = target
# --------------------------------------------------------------------------- #
def solve_wet_paper(H, dry_cols: Sequence[int], target, max_weight: int = 2):
    dry = np.asarray(list(dry_cols), dtype=np.int64)
    if dry.size == 0:
        return None
    target = target.astype(np.uint8)
    n = H.shape[1]
    for ci in dry:
        if np.array_equal(H[:, ci], target):
            e = np.zeros(n, np.uint8); e[ci] = 1; return e
    if max_weight >= 2 and dry.size >= 2:
        rng = np.random.default_rng(int(np.random.randint(0, 1 << 30)))
        order = rng.permutation(dry.size)
        limit = min(dry.size, 600)
        for a in range(limit):
            for b in range(limit):
                if a == b:
                    continue
                ca, cb = int(dry[order[a]]), int(dry[order[b]])
                if np.array_equal((H[:, ca] ^ H[:, cb]).astype(np.uint8), target):
                    e = np.zeros(n, np.uint8); e[ca] = 1; e[cb] = 1; return e
    sol = gauss_solve_GF2([H[:, ci] for ci in dry], target)
    if sol is None:
        return None
    e = np.zeros(n, np.uint8)
    for k, ci in enumerate(dry):
        if sol[k]:
            e[ci] = 1
    return e


def gauss_solve_GF2(cols, b):
    m = len(cols)
    p = len(cols[0])
    if m == 0:
        return None
    A = np.hstack([c[:, None].astype(np.uint8) for c in cols])  # (p, m)
    b = np.array(b, np.uint8).reshape(-1)
    aug = np.hstack([A, b[:, None]])
    col_piv = np.zeros(m, np.int64)
    piv_row = 0
    for c in range(m):
        r = piv_row
        while r < p and aug[r, c] == 0:
            r += 1
        if r == p:
            continue
        aug[[piv_row, r]] = aug[[r, piv_row]]
        for rr in range(p):
            if rr != piv_row and aug[rr, c]:
                aug[rr] ^= aug[piv_row]
        col_piv[piv_row] = c
        piv_row += 1
        if piv_row == p:
            break
    for r in range(piv_row, p):
        if aug[r, m] == 1 and np.all(aug[r, :m] == 0):
            return None
    x = np.zeros(m, np.uint8)
    for r in range(piv_row):
        x[col_piv[r]] = aug[r, m]
    return x


# --------------------------------------------------------------------------- #
#  载体收集：非零 AC 系数（块内直流 i%64==0 排除），按扁平序（块→块内）排序
# --------------------------------------------------------------------------- #
def _carrier_indices(y_flat: np.ndarray) -> np.ndarray:
    good = (np.arange(y_flat.size) % 64 != 0) & (y_flat != 0)
    return np.flatnonzero(good).astype(np.int64)


# --------------------------------------------------------------------------- #
#  单块伴随式编码：F5 减幅 + 湿纸，无收缩
# --------------------------------------------------------------------------- #
def _embed_block(c: np.ndarray, block_pos, m, H):
    p = H.shape[0]
    n = H.shape[1]
    xv = c[block_pos].astype(np.int64)
    xl = (xv & 1).astype(np.uint8)
    s = syndrome(H, xl)
    if np.array_equal(s, m):
        return
    d = (s ^ m).astype(np.uint8)
    tc = int(np.where(np.all(H == d[:, None], axis=0))[0][0])
    if abs(xv[tc]) > 1:
        c[block_pos[tc]] = (xv[tc] - np.sign(xv[tc])).astype(np.int16)
        return
    dry = [j for j in range(n) if abs(xv[j]) > 1]
    e = solve_wet_paper(H, dry, d, max_weight=2)
    if e is not None:
        for j in np.where(e == 1)[0]:
            v = xv[j]
            c[block_pos[j]] = (v - np.sign(v)).astype(np.int16)
        return
    c[block_pos[tc]] = (2 * np.sign(xv[tc])).astype(np.int16)


def _extract_block(c: np.ndarray, block_pos, H):
    xl = (c[block_pos] & 1).astype(np.uint8)
    return syndrome(H, xl)


# --------------------------------------------------------------------------- #
#  池内嵌 / 取：在 carriers 的一段（池）内，按池内置换分块连续埋入比特
# --------------------------------------------------------------------------- #
def _embed_pool(c, cells, pool_perm, bits, p, H):
    """cells: 该池覆盖的载体（已按扁平序排序）；pool_perm: 池内置换；就地改 c。"""
    n = (1 << p) - 1
    nb = (bits.size + p - 1) // p
    M = cells.size
    for bi in range(min(nb, M // n)):
        blk = cells[pool_perm[bi * n:(bi + 1) * n]]
        _embed_block(c, blk, bits[bi * p:(bi + 1) * p], H)


def _extract_pool(c, cells, pool_perm, num_bits, p, H) -> np.ndarray:
    n = (1 << p) - 1
    nb = (num_bits + p - 1) // p
    M = cells.size
    out = np.zeros(num_bits, np.uint8)
    for bi in range(min(nb, M // n)):
        blk = cells[pool_perm[bi * n:(bi + 1) * n]]
        seg = _extract_block(c, blk, H)
        o = out[bi * p:(bi + 1) * p]
        o[:seg.size] = seg[:o.size]
    return out


# --------------------------------------------------------------------------- #
#  认证头 / 池划分
# --------------------------------------------------------------------------- #
def cover_hash_of(chroma_flat: np.ndarray) -> bytes:
    return hashlib.sha256(np.ascontiguousarray(chroma_flat).tobytes()) \
        .digest()[:COVER_HASH_BYTES]


def head_units(p: int) -> int:
    """头部池大小 = 恰可放 COVER_HASH_BYTES*8 比特（及其 p 对齐）的载体块集合。"""
    hb = max(2, int(np.ceil(COVER_HASH_BYTES * 8 / p)))
    return hb * ((1 << p) - 1)


# --------------------------------------------------------------------------- #
#  高层：面向 YCC 量化系数对象
# --------------------------------------------------------------------------- #
def embed_into_y(y: np.ndarray, cb: np.ndarray, cr: np.ndarray, msg: str,
                 p: int = 3, password: str = "") -> tuple:
    """在 Y 量化系数上嵌入 ASCII 消息。y/cb/cr: (·,·,8,8) int16。

    返回 (new_y, report)：new_y 为嵌入后的 Y 网格，report 含 cover_hash / 容量统计。
    """
    body_bits = encode_string(msg)
    if body_bits.size == 0:
        raise ValueError("消息为空")
    H = build_hamming(p)
    n = (1 << p) - 1
    chroma = np.concatenate([np.asarray(cb).reshape(-1), np.asarray(cr).reshape(-1)])
    cover_hash = cover_hash_of(chroma)
    out = np.asarray(y).copy()
    c = out.reshape(-1)
    car = _carrier_indices(c)

    N_h = head_units(p)
    avail = car.size
    if avail < N_h + n:
        raise CapacityError(f"图像太小：可用非零AC载体 {avail}，头部+正文至少需要 {N_h + n}")
    head_pool, body_pool = car[:N_h], car[N_h:]

    # —— 头部池：seed0 = 仅口令 ——
    head_arr = np.unpackbits(np.frombuffer(cover_hash, np.uint8))
    head_pad = (-head_arr.size) % p
    head_full = np.pad(head_arr, (0, head_pad))
    _embed_pool(c, head_pool, permute_index(head_pool.size, derive_seed(b"", password)),
                head_full, p, H)

    # —— 正文池：seed = cover_hash + 口令（不足 p 的尾块补齐，提取只取真实长度）——
    body_pad = (-body_bits.size) % p
    body_full = np.pad(body_bits, (0, body_pad))
    need_body = (body_bits.size + p - 1) // p
    if body_pool.size < need_body * n:
        raise CapacityError(f"正文载体不足：需要 {need_body * n}，仅有 {body_pool.size}")
    _embed_pool(c, body_pool, permute_index(body_pool.size, derive_seed(cover_hash, password)),
                body_full, p, H)

    changed = int(np.sum(out != np.asarray(y)))
    report = dict(cover_hash=cover_hash.hex(), head_pool=N_h,
                  body_pool=int(body_pool.size), carriers_changed=changed,
                  capacity_bits=int(avail // n * p))
    return out, report


def extract_from_y(y: np.ndarray, cb: np.ndarray, cr: np.ndarray,
                   p: int = 3, password: str = "") -> tuple:
    """提取并验证。返回 (message, cover_hash_hex, tampered, head_match)。
    tampered=True 表示色度哈希失配或正文校验失败（检测到篡改）。"""
    H = build_hamming(p)
    n = (1 << p) - 1
    c = np.asarray(y).reshape(-1)
    car = _carrier_indices(c)
    N_h = head_units(p)
    if car.size < N_h:
        return None, None, True, False
    head_pool, body_pool = car[:N_h], car[N_h:]

    # 取回头部 cover_hash
    head_bits = _extract_pool(c, head_pool,
                              permute_index(head_pool.size, derive_seed(b"", password)),
                              COVER_HASH_BYTES * 8, p, H)
    if head_bits.size < COVER_HASH_BYTES * 8:
        return None, None, True, False
    cover_hash = bytes(np.packbits(head_bits))

    # 重算色度哈希 → 篡改感知
    reco = cover_hash_of(np.concatenate([np.asarray(cb).reshape(-1),
                                         np.asarray(cr).reshape(-1)]))
    head_match = reco == cover_hash

    # 正文：先读 16 bit 长度头（ceil(16/p) 个块）
    if body_pool.size < n:
        return None, cover_hash.hex(), not head_match, head_match
    permB = permute_index(body_pool.size, derive_seed(cover_hash, password))
    bit16 = _extract_pool(c, body_pool, permB, MSG_HEADER_BITS, p, H)
    length = sum(int(bit16[i]) << i for i in range(MSG_HEADER_BITS))
    total_bits = MSG_HEADER_BITS + length * 8
    need_b = (total_bits + p - 1) // p
    if body_pool.size < need_b * n:
        return None, cover_hash.hex(), True, head_match
    body = _extract_pool(c, body_pool, permB, total_bits, p, H)
    try:
        msg = decode_string(body)
    except ValueError:
        return None, cover_hash.hex(), True, head_match
    return msg, cover_hash.hex(), not head_match, head_match