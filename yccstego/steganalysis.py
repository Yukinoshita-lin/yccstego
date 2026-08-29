"""盲隐写分析 —— 面向 Y 亮度量化 DCT 系数（nsF5/F5）与重构像素图（LSB 类）。

对本压缩域工具同时提供两类信号：
  1) `analyze_y(y, cb, cr)`：直接分析量化 Y 的 AC 系数（卡方配对 + AC 系数 LSB 统计 +
     系数一阶数字规律），贴合 nsF5 在 DCT 域造成的统计扰动。
  2) `analyze(rgb)`：像素域启发式（卡方 / RS / 内容本底随机度），移植自像素版，可用于
     对比/佐证。

说明
----
盲隐写分析无 cover 原图时本质是启发式：nsF5 单次平均只改动约 p/2^p 比例的载体系数，
弱密度嵌入对统计的扰动很小；对"本底 LSB 已随机"的照片无法可靠检出。本模块组合多个
统计量，输出 0..1 的隐写倾向概率做判读参考，并诚实 abstain（无法可靠判定）。
"""

from __future__ import annotations
import math

import numpy as np

# --------------------------------------------------------------------------- #
#  通用数学工具
# --------------------------------------------------------------------------- #
def _gser(a, x, itmax=200, eps=3e-14):
    if a <= 0:
        return 1.0
    if x <= 0:
        return 0.0
    ap, s, del_ = a, 1.0 / a, 1.0 / a
    for _ in range(itmax):
        ap += 1.0
        del_ *= x / ap
        s += del_
        if abs(del_) < abs(s) * eps:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gcf(a, x, itmax=200, eps=3e-14, fpmin=1e-300):
    if a <= 0:
        return 0.0
    b = x + 1.0 - a
    c, d, h = 1.0 / fpmin, 1.0 / b, 1.0 / b
    for i in range(1, itmax + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        del_ = d * c
        h *= del_
        if abs(del_ - 1.0) < eps:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def gamma_q(a, x):
    return 1.0 - _gser(a, x) if x < a + 1.0 else _gcf(a, x)


def chi2_sf(x, df):
    """卡方生存函数 1-CDF。"""
    return gamma_q(df / 2.0, x / 2.0)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# ========================================================================== #
#  像素域（重构图 / 常规图像）
# ========================================================================== #
def _channel(image):
    img = np.ascontiguousarray(image)
    return img[..., 0] if img.ndim == 3 else img


def _counts(image):
    return np.bincount(_channel(image).ravel(), minlength=256).astype(np.float64)


def chi2_stats(counts: np.ndarray):
    even = counts[0::2]; odd = counts[1::2]
    sums = even + odd
    mask = sums > 0
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0
    if n <= 1:
        return float(np.sum((even[mask] - odd[mask]) ** 2 / sums[mask])), 1.0, 0
    stat = float(np.sum((even[mask] - odd[mask]) ** 2 / sums[mask]))
    return stat, float(chi2_sf(stat, n - 1)), n - 1


def diff_entropy(image, axis_bins=256):
    ch = _channel(image).astype(np.int16)
    d = np.concatenate([np.abs(np.diff(ch, axis=1)).ravel(),
                        np.abs(np.diff(ch, axis=0)).ravel()])
    hist = np.bincount(np.clip(d, 0, axis_bins - 1), minlength=axis_bins).astype(np.float64)
    hist = hist / hist.sum()
    nz = hist[hist > 0]
    return float(-np.sum(nz * np.log2(nz)))


def rs_metrics(image, mask=np.array([0, 1, 1, 0])):
    ch = _channel(image)
    flat = ch.ravel()
    m = len(flat) - len(flat) % 4
    groups = flat[:m].reshape(-1, 4)
    n = groups.shape[0]
    if n == 0:
        return {"Rn": 0, "Sn": 0, "Gr": 0.0, "Gn": 0.0}
    gi = groups.astype(np.int16)
    fg = np.abs(np.diff(gi, axis=1)).sum(axis=1)
    pos = ((gi ^ mask) & 0xFF).astype(np.int16)
    neg = ((gi ^ (1 - mask)) & 0xFF).astype(np.int16)
    fM = np.abs(np.diff(pos, axis=1)).sum(axis=1)
    fN = np.abs(np.diff(neg, axis=1)).sum(axis=1)
    Rm = int(np.sum(fM > fg)); Sm = int(np.sum(fM < fg))
    Rn = int(np.sum(fN > fg)); Sn = int(np.sum(fN < fg))
    return {"Rn": Rn, "Sn": Sn, "Gr": (Rm - Sm) / n, "Gn": (Rn - Sn) / n}


def lsb_diff_entropy(image):
    lsb = _channel(image).astype(np.int16) & 1
    d = np.concatenate([np.abs(np.diff(lsb, axis=1)).ravel(),
                        np.abs(np.diff(lsb, axis=0)).ravel()])
    hist = np.bincount(d, minlength=2).astype(np.float64)
    hist = hist / hist.sum()
    nz = hist[hist > 0]
    return float(-np.sum(nz * np.log2(nz)))


# --------------------------------------------------------------------------- #
#  Y 量化 DCT 系数的分析（本工具主信号）
# --------------------------------------------------------------------------- #
def _ac_coeffs(y):
    """量化 Y 网格的全部 AC 系数（去直流、去零），返回一维 int64。"""
    flat = np.asarray(y).reshape(-1)
    return flat[(np.arange(flat.size) % 64 != 0) & (flat != 0)].astype(np.int64)


def dct_ac_chisquare(ac_coeffs: np.ndarray):
    """Westfeld 式卡方：AC 系数成对 (2k, 2k+1) 在隐写后趋近均衡。返回 (stat, p, df)。

    nsF5/F5 通过翻 LSB 把奇数幅值 +-1 移向偶数幅值，造成相邻幅值计数趋同。
    """
    mag = np.abs(ac_coeffs)
    hi = int(mag.max()) if mag.size else 0
    hi = max(hi + 1, 64)
    counts = np.bincount(mag, minlength=hi + 1).astype(np.float64)
    if counts.size % 2 == 1:             # 保证偶数个桶以配对 (2k,2k+1)
        counts = np.append(counts, 0.0)
    return chi2_stats(counts)


def _plane_diff_entropy2(y, axis_bins=32):
    """量化 AC 系数网格在空间块排布上的相邻差分熵，衡量系数 LSB 结构的本底随机度。"""
    ac = np.asarray(y).astype(np.int64)
    ac[:, :, 0, 0] = 0                       # 去掉直流影响
    flat = ac.reshape(ac.shape[0] * 8, ac.shape[1] * 8)[:, :]
    d = np.concatenate([np.abs(np.diff(flat, axis=1)).ravel(),
                        np.abs(np.diff(flat, axis=0)).ravel()])
    hist = np.bincount(np.clip(d, 0, axis_bins - 1), minlength=axis_bins).astype(np.float64)
    hist = hist / hist.sum()
    nz = hist[hist > 0]
    return float(-np.sum(nz * np.log2(nz)))


_SENS = {
    "严格 (低误报)": ((0.50, 0.80), 0.85),
    "均衡":           ((0.40, 0.70), 1.00),
    "宽松 (高检出)": ((0.30, 0.60), 1.18),
}


def analyze_y(y, sensitivity="均衡", unit_baseline=None):
    """在量化 Y 系数上做盲隐写分析。

    主信号：magnitude-1 占比（|c|==1 的非零 AC 占比）。nsF5/F5 把 |c|==2 的载体系数
    减幅到 |c|==1（湿阈值），因此该占比随嵌入密度单调上升，是 F5/nsF5 的特征指纹。
    提供 unit_baseline（干净覆盖的该占比）可得到随密度上升的检测曲线；缺省时用名义
    基线并加入系数本底随机度作出可判性约束（弱嵌入本底随机时诚实 abstain）。
    """
    ac = _ac_coeffs(y)
    if ac.size == 0:
        return {"ok": False, "n_ac": 0}
    parity = float(np.mean((ac & 1).astype(np.float64)))
    unit_frac = float(np.mean((np.abs(ac) == 1).astype(np.float64)))
    chi_stat, chi_p, df = dct_ac_chisquare(ac)
    h = _plane_diff_entropy2(y)

    # 系数本底随机度：低→结构强可判；高→天然抖动/噪声
    ac_rand = float(np.clip((h - 1.0) / (3.5 - 1.0), 0.0, 1.0))
    struct = 1.0 - ac_rand

    base = unit_baseline if unit_baseline is not None and unit_baseline > 0 else 0.40
    gain = max(0.0, (unit_frac - base) / max(base, 1e-3))
    s_unit = np.clip(gain / 0.35, 0, 1) * struct  # 增长 35% 即视为强信号

    # 卡方对 AC 幅值直方图不适用（天然高度非均匀），仅作弱佐证
    s_chi = np.clip((chi_p - 0.9) / 0.05, 0, 1) * struct * 0.3
    raw = max(s_unit, s_chi)

    (thr_poss, thr_high), pull_c = _SENS.get(sensitivity, _SENS["均衡"])

    if ac_rand >= 0.55:
        prob = float(np.clip(0.5 + 0.15 * raw, 0.5, 0.62))
        abstain = "AC 系数本底随机"
    else:
        prob = float(np.clip(0.05 + 0.95 * raw, 0.0, 1.0))
        abstain = None
    prob = float(np.clip(0.5 + (prob - 0.5) * pull_c, 0.0, 1.0))

    if abstain is not None:
        verdict = f"无法可靠判定({abstain})"
    else:
        verdict = ("高度可能被隐写" if prob >= thr_high else
                   "可能被隐写" if prob >= thr_poss else
                   "不太可能被隐写")
    return {
        "ok": True, "n_ac": int(ac.size),
        "parity_odd": round(parity, 4),
        "unit_frac": round(unit_frac, 4), "unit_baseline": round(base, 4),
        "dct_chi2_pvalue": round(chi_p, 4), "ac_plane_entropy": round(h, 4),
        "stego_probability_dct": prob, "verdict": verdict,
    }


def analyze(image, base_gn=0.62, sensitivity="均衡"):
    """像素域启发式分析（Reconstructed RGB / 常规位图）。"""
    cnt = _counts(image)
    chi_stat, chi_p, df = chi2_stats(cnt)
    rs = rs_metrics(image)

    flat = _channel(image).ravel()
    steps = 20
    ps = []
    for i in range(1, steps + 1):
        end = max(64, int(flat.size * i / steps))
        c = np.bincount(flat[:end], minlength=256).astype(np.float64)
        _, p, _ = chi2_stats(c)
        ps.append(p if p is not None else 0.0)
    median_p = float(np.median(ps)) if ps else 0.0

    h = diff_entropy(image)
    lg = lsb_diff_entropy(image)
    tex = float(np.clip((h - 1.2) / (7.0 - 1.2), 0.0, 1.0))
    Gn = rs["Gn"]; Gr = rs["Gr"]

    gn_floor = 0.06 * tex
    baseline = base_gn * (1.0 - 0.8 * tex) + gn_floor
    drop = max(0.0, (baseline - Gn) / max(baseline, 1e-3))
    center = 0.34 + 0.20 * tex
    s_rs = _sigmoid((drop - center) / 0.12)
    struct = 1.0 - tex
    s_chi = np.clip((median_p - 0.35) / 0.5, 0, 1) * pow(struct, 2)
    est_rate = float(np.clip(0.55 * max(0.0, (0.62 - Gn) / 0.62) +
                             0.45 * max(0.0, (0.72 - Gr) / 0.72), 0.0, 1.0))
    raw = max(s_rs, s_chi)
    lsb_rand = float(np.clip((lg - 0.90) / 0.08, 0.0, 1.0))
    (thr_poss, thr_high), pull_c = _SENS.get(sensitivity, _SENS["均衡"])

    if tex >= 0.55:
        prob = float(np.clip(0.5 + 0.10 * raw, 0.5, 0.62)); _abstain = "图像本底噪声高"
    elif lsb_rand > 0.0:
        prob = float(np.clip(0.5 + 0.12 * lsb_rand * raw, 0.5, 0.62)); _abstain = "LSB 位平面已随机化"
    else:
        prob = float(np.clip(0.04 + 0.96 * raw, 0.0, 1.0)); _abstain = None
        if tex < 0.35 and (Gn / max(base_gn, 1e-3)) > 0.82 and median_p < 0.1:
            prob = min(prob, 0.15)
    prob = float(np.clip(0.5 + (prob - 0.5) * pull_c, 0.0, 1.0))

    if _abstain is not None:
        verdict = f"无法可靠判定({_abstain})"
    else:
        verdict = ("高度可能被隐写" if prob >= thr_high else
                   "可能被隐写" if prob >= thr_poss else
                   "不太可能被隐写")
    return {
        "chi2_stat": chi_stat, "chi2_pvalue": chi_p, "median_prefix_p": median_p,
        "diff_entropy": h, "lsb_diff_entropy": lg, "texture_noise": tex,
        "RS_Gr": Gr, "RS_Gn": Gn, "est_rate": est_rate,
        "stego_probability": prob, "verdict": verdict, "sensitivity": sensitivity,
        "prefix_ps": [float(x) for x in ps],
    }