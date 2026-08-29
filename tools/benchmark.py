"""yccstego 基准评测：往返保真 / 容量 / 安全（隐写可检测性 vs 嵌入密度）。

输出 CSV（默认 F:/Steganography/yccstego/output/bench_yccstego.csv）：
  行维度 = (p, quality, 消息长度)。列 =
    msg_chars, body_pool, capacity_bits, cap_used_bits, density,
    carriers_changed, changed_rate, embedding_efficiency(=payload/changed),
    ber, dct_detect_prob, parity_odd, pixels_ok(重构无异常)。

运行：python tools/benchmark.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yccstego import jpeg_codec as J
from yccstego import nsf5
from yccstego.api import embed_bytes, extract_bytes, analyze_bytes


IMG_SIZES = [(160, 192), (256, 256)]
P_LIST = [1, 2, 3, 4, 5]
Q_LIST = [70, 85, 95]
MSG_LENS = [16, 64, 256, 1024, 4096]


def make_rgb(h, w, seed, texture=1.0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, w)[None, :]
    yy = np.linspace(0, 1, h)[:, None]
    base = np.sin(3 * x) * np.cos(5 * yy) * 60 + 120
    R = np.clip(base + 3 * x * 255 * np.random.uniform(0.5, 1.5, (h, w)), 0, 255)
    G = np.clip(120 + 60 * np.sin(8 * x) * np.exp(-(yy - 0.5) ** 2 * 20)
                + rng.normal(0, 8 * texture, (h, w)), 0, 255)
    B = np.clip(base - 40 * x + rng.normal(0, 8 * texture, (h, w)), 0, 255)
    return np.stack([R, G, B], -1).astype(np.uint8)


def msg_of(n):
    return "M" * n  # 全部 ASCII 可表示


def _unit_frac(y):
    flat = np.asarray(y).reshape(-1)
    ac = flat[(np.arange(flat.size) % 64 != 0) & (flat != 0)]
    return float(np.mean(np.abs(ac) == 1)) if ac.size else 0.0


def run_case(rgb, p, quality, n):
    # 覆盖 JPEG 的量化系数网格
    c0 = J.YCC(rgb, quality)
    chroma = np.concatenate([c0.Cb.reshape(-1), c0.Cr.reshape(-1)])
    cover_hash = nsf5.cover_hash_of(chroma)
    H = nsf5.build_hamming(p)
    body_pool = int(nsf5._carrier_indices(c0.Y.reshape(-1)).size - nsf5.head_units(p))
    capacity_bits = int(body_pool // ((1 << p) - 1) * p)

    payload = nsf5.encode_string(msg_of(n))
    if payload.size > capacity_bits:
        return None
    density = payload.size / max(capacity_bits, 1)

    stego, rep = embed_bytes(rgb, msg_of(n), p=p, password="", quality=quality)
    # 往返保真：提取到比特级一致
    out, cov, tampered, hm = extract_bytes(stego, p=p, password="")
    ber = 0.0 if (out == msg_of(n)) else 1.0
    changed = rep["carriers_changed"]
    eff = payload.size / max(changed, 1)
    # 安全：用该覆盖的干净 magnitude-1 占比作基线，测隐写后占比抬升带来的检测概率
    clean_unit = _unit_frac(c0.Y)
    ana = analyze_bytes(stego, unit_baseline=clean_unit)
    return dict(
        p=p, quality=quality, size=f"{c0.w}x{c0.h}", msg_chars=n,
        body_pool=body_pool, capacity_bits=capacity_bits, cap_used_bits=int(payload.size),
        density=round(density, 4), carriers_changed=changed,
        changed_rate=round(changed / max(rep["body_pool"], 1), 6),
        embedding_efficiency=round(eff, 4), ber=ber, tampered=int(tampered),
        dct_detect_prob=round(ana["stego_probability_dct"], 4),
        unit_frac=ana.get("unit_frac"), clean_unit=round(clean_unit, 4),
        n_ac=ana.get("n_ac"),
    )


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_yccstego.csv")
    fields = ["p", "quality", "size", "msg_chars", "body_pool", "capacity_bits",
              "cap_used_bits", "density", "carriers_changed", "changed_rate",
              "embedding_efficiency", "ber", "tampered", "dct_detect_prob",
              "unit_frac", "clean_unit", "n_ac"]
    rows = []
    for (h, w) in IMG_SIZES:
        rgb = make_rgb(h, w, seed=h * w % 8191)
        for q in Q_LIST:
            for p in P_LIST:
                for n in MSG_LENS:
                    r = run_case(rgb, p, q, n)
                    status = "ok" if r else "skip(超容量)"
                    print(f"  p={p} q={q} {w}x{h} n={n}: {status}")
                    if r:
                        rows.append(r)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\n已写出：{out_path}  （{len(rows)} 行）")
    # 摘要：嵌入效率与密度-检测率
    if rows:
        for p in P_LIST:
            sub = [r for r in rows if r["p"] == p and r["quality"] == 85 and r["density"] <= 0.5]
            if sub:
                eff = np.mean([r["embedding_efficiency"] for r in sub])
                det = {r["msg_chars"]: r["dct_detect_prob"]
                       for r in rows if r["p"] == p and r["quality"] == 85 and r["density"] > 0}
                print(f"p={p}: 平均嵌入效率={eff:.3f}  密度→检测率映射={det}")


if __name__ == "__main__":
    main()