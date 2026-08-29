"""C++ DLL vs Python 的 Y 量化 DCT 特征一致性对比。

对一批图像(覆盖 + 不同密度的 nsF5 隐写)分别用:
  - yccstego.steganalysis 的原语(_ac_coeffs / dct_ac_chisquare / _plane_diff_entropy2)
  - yccstego/cpp/dcffeatures.dll (ctypes)
计算同一组特征(n_ac/parity/unit_frac/chi_stat/chi_p/df/ac_plane_entropy),
逐字段比对绝对误差, 输出 output/compare_cpp_python.csv 并给出结论。

运行: python tools/compare_cpp_python.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yccstego import jpeg_codec as J
from yccstego import steganalysis as S
from yccstego import cpp_features as CF

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
FIELDS = ["n_ac", "parity_odd", "unit_frac", "chi_stat", "chi_p", "df", "ac_plane_entropy"]
TOL = 1e-9  # 整数字段须零差, 浮点按此容差


def make_scene(kind, h, w, seed):
    """几类图片以覆盖不同 AC 本底: 平滑、强纹理、混合(像照片场景)。"""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 1, w)[None, :]
    y = np.linspace(0, 1, h)[:, None]
    if kind == "smooth":
        base = 90 + 60 * np.sin(2 * x) * np.cos(3 * y) + 60 * y
    elif kind == "textured":
        base = 128 + 90 * rng.standard_normal((h, w))
    else:  # mixed: 分区块加渐变, 模拟照片场景
        base = np.zeros((h, w))
        for bx in range(4):
            for by in range(4):
                c = 60 + (bx * 31 + by * 47) % 140
                base += c * ((y * 4).astype(int) == by) * ((x * 4).astype(int) == bx)
        base += 40 * np.sin(6 * x) * np.cos(5 * y) + rng.normal(0, 6, (h, w))
    R = np.clip(base + 30 * x * 255 + rng.normal(0, 10, (h, w)), 0, 255).astype(np.uint8)
    G = np.clip(base + 40 * y * 255 + rng.normal(0, 10, (h, w)), 0, 255).astype(np.uint8)
    B = np.clip(255 - base + rng.normal(0, 10, (h, w)), 0, 255).astype(np.uint8)
    return np.stack([R, G, B], -1)


def py_features(y):
    ac = S._ac_coeffs(y)
    chi_stat, chi_p, df = S.dct_ac_chisquare(ac)
    h = S._plane_diff_entropy2(y)
    return {
        "n_ac": int(ac.size),
        "parity_odd": float(np.mean((ac & 1).astype(np.float64))),
        "unit_frac": float(np.mean((np.abs(ac) == 1).astype(np.float64))),
        "chi_stat": chi_stat, "chi_p": chi_p, "df": int(df),
        "ac_plane_entropy": h,
    }


def embed_y(rgb, msg, p, quality=85):
    """返回嵌入后重解析的量化 Y 网格 (R,C,8,8)。"""
    from yccstego import api
    b, _ = api.embed_bytes(rgb, msg, p=p, password="pw", quality=quality)
    return J.YCC.from_bytes(b).Y


def compare(y, name):
    py = py_features(y)
    dl = CF.analyze_y_dll(y)
    if dl is None:
        return None
    ok = all(abs(py[f] - dl[f]) <= (0.0 if f in ("n_ac", "df") else TOL)
             for f in FIELDS)
    return {
        "name": name,
        "n_ac_py": py["n_ac"], "n_ac_dll": dl["n_ac"],
        "parity_py": round(py["parity_odd"], 6), "parity_dll": round(dl["parity_odd"], 6),
        "unit_py": round(py["unit_frac"], 6), "unit_dll": round(dl["unit_frac"], 6),
        "chi_p_py": round(py["chi_p"], 8), "chi_p_dll": round(dl["chi_p"], 8),
        "h_py": round(py["ac_plane_entropy"], 6), "h_dll": round(dl["ac_plane_entropy"], 6),
        "max_err": max(abs(py[f] - dl[f]) for f in FIELDS),
        "pass": int(ok),
    }


def main():
    if not CF.available():
        print("未找到 dcffeatures.dll，请编译: g++ -O2 -shared -o cpp/dcffeatures.dll "
              "cpp/dcffeatures.cpp -static")
        sys.exit(2)
    os.makedirs(OUT, exist_ok=True)
    rows, results = [], []
    cases = [("smooth", 1), ("textured", 2), ("mixed", 3)]
    for kind, seed in cases:
        rgb = make_scene(kind, 96, 128, seed)
        row = compare(J.YCC(rgb, 85).Y, f"{kind}_cover")
        rows.append(row); results.append(row["pass"])
        for p, msg in [(1, "a" * 300), (2, "abcdef" * 40), (3, "stego" * 50),
                       (5, "s123456789")]:
            try:
                y = embed_y(rgb, msg, p)
            except Exception:
                continue
            row = compare(y, f"{kind}_stego_p{p}")
            rows.append(row); results.append(row["pass"])

    path = os.path.join(OUT, "compare_cpp_python.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    npass, n = sum(results), len(results)
    worst = max(rows, key=lambda r: r["max_err"])
    print(f"对比 {n} 组(覆盖+隐写) 通过 {npass}/{n}; 最大/组误差={worst['max_err']:.2e} ({worst['name']})")
    if not all(results):
        for r in filter(lambda r: not r["pass"], rows):
            print("  MISMATCH:", r["name"], "max_err=", r["max_err"])
        print("结论: C++ 与 Python 特征不一致(见 CSV)"); sys.exit(1)
    print(f"结论: C++ DLL 与 Python 逐项一致(容差 {TOL}) → 同一检测逻辑")
    print(f"CSV → {path}")


if __name__ == "__main__":
    main()