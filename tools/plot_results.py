"""从 bench_yccstego.csv 生成结果图(往返保真 / 容量 / 嵌入效率 / 安全检测)。

输出 PNG 到 output/ 目录:
  fig_efficiency.png   嵌入效率 vs p(码族)  + 理论界
  fig_capacity.png     载荷容量 vs 质量/图尺寸
  fig_security.png     隐写检测率 vs 嵌入密度(q=85)
  fig_lsbcost.png      改动载体比例 vs 密度

运行: python tools/plot_results.py [csv路径]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yccstego import nsf5

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def hamming_efficiency_bound(p: float) -> float:
    """二元汉明码 [2^p-1, 2^p-1-p] 的嵌入效率理论上界 p·2^p/(2^p-1)。"""
    return p * (2 ** p) / (2 ** p - 1)


def plot_efficiency(df):
    # 用 q=85, 使变化较多的一条曲线(密度≥某阈值, 取最大填充密度的点)
    sub = df[df["quality"] == 85].copy()
    # 取每 (p) 中嵌入效率的样本均值(密度不同的平均) & p理论界
    ps = sorted(sub["p"].unique())
    effs = [sub[sub["p"] == p]["embedding_efficiency"].mean() for p in ps]
    bounds = [hamming_efficiency_bound(p) for p in ps]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ps, bounds, "k--", label="汉明码理论上界 $p\\cdot 2^p/(2^p-1)$")
    ax.plot(ps, effs, "o-", color="#d62728", label="yccstego 实测(q=85 平均)")
    ax.set_xlabel("矩阵编码参数 $p$（码族 $[2^p-1,\\,2^p-1-p]$）")
    ax.set_ylabel("嵌入效率 (bits / 载体改动)")
    ax.set_title("nsF5 嵌入效率 vs 码族($p$)")
    ax.grid(alpha=0.3)
    # 标出效率增量
    for p, e, b in zip(ps, effs, bounds):
        ax.annotate(f"{e:.2f}", (p, e), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_efficiency.png"), dpi=150)
    plt.close(fig)
    print("[1/4] fig_efficiency.png  (嵌入效率 vs 码族)")


def plot_capacity(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # 左: 最大载荷 vs 质量(取每 q 的最大 cap_used_bits 作“满嵌”代表)
    sub = df.groupby("quality")["capacity_bits"].max().reset_index()
    axes[0].bar([str(q) for q in sub["quality"]], sub["capacity_bits"],
                color="#1f77b4")
    axes[0].set_xlabel("JPEG 质量")
    axes[0].set_ylabel("最大载荷 (bits)")
    axes[0].set_title("可嵌入容量 vs 质量")
    axes[0].grid(axis="y", alpha=0.3)
    # 右: 不同图尺寸的容量(固定 q=85,p=1)
    sub2 = df[(df["quality"] == 85) & (df["p"] == 1)]
    agg = sub2.groupby("size")["capacity_bits"].max()
    xs = np.arange(len(agg))
    axes[1].bar(xs, agg.values, color=["#2ca02c", "#ff7f0e"])
    axes[1].set_xticks(xs); axes[1].set_xticklabels(agg.index)
    axes[1].set_xlabel("图尺寸 (W×H)")
    axes[1].set_ylabel("最大载荷 (bits)")
    axes[1].set_title("容量 vs 图尺寸(q=85, p=1)")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_capacity.png"), dpi=150)
    plt.close(fig)
    print("[2/4] fig_capacity.png  (容量 vs 质量/尺寸)")


def plot_security(df):
    sub = df[(df["quality"] == 85) & (df["p"].isin([1, 3, 5]))].copy()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for p in [1, 3, 5]:
        d = sub[sub["p"] == p]
        ax.plot(d["density"], d["dct_detect_prob"], "o-", label=f"p={p}")
    ax.axhline(0.05, color="gray", ls="--", lw=1, label="判定基线(地板 0.05)")
    ax.set_xlabel("嵌入密度 (载荷 bits / 容量 bits)")
    ax.set_ylabel("DCT 域隐写检测概率 (magnitude-1 指纹)")
    ax.set_title("Y 亮度 DCT 域 nsF5 可检测性 vs 嵌入密度(q=85)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_security.png"), dpi=150)
    plt.close(fig)
    print("[3/4] fig_security.png  (检测率 vs 密度)")


def plot_lsb_cost(df):
    sub = df[df["quality"] == 85].copy()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for p in sorted(sub["p"].unique()):
        d = sub[sub["p"] == p]
        ax.plot(d["density"], d["changed_rate"] * 100, "o-", label=f"p={p}")
    ax.set_xlabel("嵌入密度")
    ax.set_ylabel("载体系数改动比例 (%)")
    ax.set_title("嵌入代价：改动载体系数比例 vs 密度(q=85)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lsbcost.png"), dpi=150)
    plt.close(fig)
    print("[4/4] fig_lsbcost.png  (改动代价 vs 密度)")


def main(csv_path):
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(csv_path)
    # 校验往返保真
    assert (df["ber"] == 0).all(), "存在 BER!=0, 往返保真被破坏!"
    assert (df["tampered"] == 0).all(), "存在被感知篡改!"
    print(f"数据 {len(df)} 行: 全部 BER=0(逐位一致), tampered=0 ✓")
    plot_efficiency(df)
    plot_capacity(df)
    plot_security(df)
    plot_lsb_cost(df)
    print(f"\n图已写出到: {OUT}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "bench_yccstego.csv")
    main(path)